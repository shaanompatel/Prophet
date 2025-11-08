use std::{
    collections::HashMap,
    env,
    fs,
    path::Path,
    sync::Mutex,
};

use dotenvy::dotenv;
use anyhow::Result;
use futures::{SinkExt, StreamExt};
use once_cell::sync::Lazy;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio_tungstenite::connect_async;
use tungstenite::Message;

use borsh::{BorshDeserialize, BorshSerialize};
use solana_client::nonblocking::rpc_client::RpcClient;
use solana_sdk::{
    commitment_config::CommitmentConfig,
    instruction::{AccountMeta, Instruction},
    program_pack::Pack,
    pubkey::Pubkey,
    signature::{Keypair, Signer},
    system_instruction,
    transaction::Transaction,
};
use spl_token::{
    id as spl_token_program_id,
    instruction as token_instruction,
    state::{Account as TokenAccount, Mint},
};

const ACTION_WS_URL: &str = "ws://localhost:8766";

/// Must match on-chain MARKET_SEED
const MARKET_SEED: &[u8] = b"market";

/// One AI action from decide_markets.py
#[derive(Debug, Deserialize)]
struct Action {
    action: String,
    market_name: String,
    probability: Option<f64>,
    outcome: Option<String>,
    reason: Option<String>,
}

/// The instruction enum layout MUST match the on-chain program.
#[derive(BorshSerialize, BorshDeserialize, Debug, Clone)]
pub enum ProphetInstruction {
    InitializeMarket {
        event_id: [u8; 32],
        category: u8,
        end_timestamp: i64,
    },
    BuyYes {
        amount: u64,
    },
    SellYes {
        amount: u64,
    },
    BuyNo {
        amount: u64,
    },
    SellNo {
        amount: u64,
    },
    ResolveMarket {
        outcome: u8,
    },
}

/// Info we keep in memory per market so RESOLVE can find it.
struct MarketInfo {
    event_id: [u8; 32],
    market_pubkey: Pubkey,
    yes_mint: Pubkey,
    no_mint: Pubkey,
    yes_vault: Pubkey,
    no_vault: Pubkey,
}

static MARKET_REGISTRY: Lazy<Mutex<HashMap<String, MarketInfo>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[tokio::main]
async fn main() -> Result<()> {
    // Load variables from .env if present
    dotenv().ok();

    // --- Load env config ---
    let program_id: Pubkey = env::var("PROPHET_PROGRAM_ID")?
        .parse()
        .expect("Invalid PROPHET_PROGRAM_ID");

    let rpc_url = env::var("SOLANA_RPC_URL")
        .unwrap_or_else(|_| "https://api.devnet.solana.com".to_string());

    let keypair_path = dirs::home_dir()
        .expect("no home dir")
        .join(".config/solana/id.json");
    let payer = read_keypair(&keypair_path)?;

    let client = RpcClient::new_with_commitment(rpc_url, CommitmentConfig::confirmed());

    println!("[bridge] Wallet:  {}", payer.pubkey());
    println!("[bridge] Program: {}", program_id);
    println!("[bridge] RPC:     {}", client.url());

    // --- Connect to AI Action WebSocket ---
    println!("[bridge] Connecting to {}", ACTION_WS_URL);
    let (ws_stream, _) = connect_async(ACTION_WS_URL).await?;
    println!("[bridge] Connected to AI action server");

    let (mut ws_write, mut ws_read) = ws_stream.split();

    while let Some(msg) = ws_read.next().await {
        let msg = msg?;
        if msg.is_text() {
            let text = msg.to_text()?;
            match serde_json::from_str::<Action>(text) {
                Ok(action) => {
                    println!("\n[bridge] Received Action: {:?}", action);

                    match action.action.as_str() {
                        "CREATE" => {
                            handle_create(&client, &payer, &program_id, &action).await?;
                            // Optional ack back to AI
                            ws_write
                                .send(Message::Text(
                                    r#"{"status":"ok","type":"CREATE"}"#.into(),
                                ))
                                .await?;
                        }
                        "RESOLVE" => {
                            handle_resolve(&client, &payer, &program_id, &action).await?;
                            ws_write
                                .send(Message::Text(
                                    r#"{"status":"ok","type":"RESOLVE"}"#.into(),
                                ))
                                .await?;
                        }
                        other => {
                            eprintln!("[bridge] Unknown action type: {}", other);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("[bridge] Failed to parse JSON action: {e}. Raw: {text}");
                }
            }
        }
    }

    Ok(())
}

/// Read a solana-keygen JSON keypair file.
fn read_keypair(path: &Path) -> Result<Keypair> {
    let data = fs::read_to_string(path)?;
    let bytes: Vec<u8> = serde_json::from_str(&data)?;
    let kp = Keypair::from_bytes(&bytes)?;
    Ok(kp)
}

/// Hash a market name to a 32-byte event_id (must match on-chain convention).
fn event_id_from_name(name: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(name.as_bytes());
    let hash = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&hash[..32]);
    out
}

// --------------------------------------------------------
// CREATE handler: full on-chain setup + InitializeMarket
// --------------------------------------------------------

async fn handle_create(
    client: &RpcClient,
    payer: &Keypair,
    program_id: &Pubkey,
    action: &Action,
) -> Result<()> {
    let market_name = &action.market_name;
    let event_id = event_id_from_name(market_name);

    // Derive the PDA for this market (must match on-chain logic)
    let (market_pda, _bump) =
        Pubkey::find_program_address(&[MARKET_SEED, &event_id], program_id);

    println!(
        "[bridge] Creating market '{}' with event_id {:?} (PDA = {})",
        market_name,
        &event_id[..4],
        market_pda
    );

    // New accounts for mints and vaults
    let yes_mint_kp = Keypair::new();
    let no_mint_kp = Keypair::new();
    let yes_vault_kp = Keypair::new();
    let no_vault_kp = Keypair::new();

    // Rent-exempt balances for SPL accounts
    let mint_rent = client
        .get_minimum_balance_for_rent_exemption(Mint::LEN)
        .await?;
    let ata_rent = client
        .get_minimum_balance_for_rent_exemption(TokenAccount::LEN)
        .await?;

    // ---------- Transaction 1: create + init mints and vaults ----------
    let mut ixs = vec![];

    // YES mint
    ixs.push(system_instruction::create_account(
        &payer.pubkey(),
        &yes_mint_kp.pubkey(),
        mint_rent,
        Mint::LEN as u64,
        &spl_token_program_id(),
    ));

    // NO mint
    ixs.push(system_instruction::create_account(
        &payer.pubkey(),
        &no_mint_kp.pubkey(),
        mint_rent,
        Mint::LEN as u64,
        &spl_token_program_id(),
    ));

    // YES vault token account
    ixs.push(system_instruction::create_account(
        &payer.pubkey(),
        &yes_vault_kp.pubkey(),
        ata_rent,
        TokenAccount::LEN as u64,
        &spl_token_program_id(),
    ));

    // NO vault token account
    ixs.push(system_instruction::create_account(
        &payer.pubkey(),
        &no_vault_kp.pubkey(),
        ata_rent,
        TokenAccount::LEN as u64,
        &spl_token_program_id(),
    ));

    // Initialize YES mint (6 decimals)
    ixs.push(token_instruction::initialize_mint(
        &spl_token_program_id(),
        &yes_mint_kp.pubkey(),
        &payer.pubkey(), // mint authority
        None,
        6,
    )?);

    // Initialize NO mint
    ixs.push(token_instruction::initialize_mint(
        &spl_token_program_id(),
        &no_mint_kp.pubkey(),
        &payer.pubkey(),
        None,
        6,
    )?);

    // Initialize YES vault account (token account owner = Market PDA)
    ixs.push(token_instruction::initialize_account(
        &spl_token_program_id(),
        &yes_vault_kp.pubkey(),
        &yes_mint_kp.pubkey(),
        &market_pda, // <-- PDA is vault owner
    )?);

    // Initialize NO vault account (token account owner = Market PDA)
    ixs.push(token_instruction::initialize_account(
        &spl_token_program_id(),
        &no_vault_kp.pubkey(),
        &no_mint_kp.pubkey(),
        &market_pda, // <-- PDA is vault owner
    )?);

    // Mint initial liquidity into both vaults (1,000 units each)
    let initial_liquidity: u64 = 1_000 * 10u64.pow(6); // 1000 * 10^6

    ixs.push(token_instruction::mint_to(
        &spl_token_program_id(),
        &yes_mint_kp.pubkey(),
        &yes_vault_kp.pubkey(),
        &payer.pubkey(),
        &[],
        initial_liquidity,
    )?);

    ixs.push(token_instruction::mint_to(
        &spl_token_program_id(),
        &no_mint_kp.pubkey(),
        &no_vault_kp.pubkey(),
        &payer.pubkey(),
        &[],
        initial_liquidity,
    )?);

    let recent = client.get_latest_blockhash().await?;
    let signers: [&Keypair; 5] = [
        payer,
        &yes_mint_kp,
        &no_mint_kp,
        &yes_vault_kp,
        &no_vault_kp,
    ];

    let tx = Transaction::new_signed_with_payer(
        &ixs,
        Some(&payer.pubkey()),
        &signers[..],
        recent,
    );

    let sig = client.send_and_confirm_transaction(&tx).await?;
    println!("[bridge] Mint & vault setup tx: {}", sig);

    // ---------- Transaction 2: call InitializeMarket on your program ----------

    let category: u8 = 0;        // e.g. 0 = generic
    let end_timestamp: i64 = 0;  // you can choose to set a real expiry later

    let init_ix_data = ProphetInstruction::InitializeMarket {
        event_id,
        category,
        end_timestamp,
    }
    .try_to_vec()?;

    let init_ix = Instruction {
        program_id: *program_id,
        accounts: vec![
            AccountMeta::new(market_pda, false),
            AccountMeta::new_readonly(payer.pubkey(), true),
            AccountMeta::new_readonly(yes_mint_kp.pubkey(), false),
            AccountMeta::new_readonly(no_mint_kp.pubkey(), false),
            AccountMeta::new(yes_vault_kp.pubkey(), false),
            AccountMeta::new(no_vault_kp.pubkey(), false),
            AccountMeta::new_readonly(spl_token_program_id(), false),
            AccountMeta::new_readonly(solana_sdk::system_program::id(), false),
        ],
        data: init_ix_data,
    };

    let recent2 = client.get_latest_blockhash().await?;
    let signers2: [&Keypair; 1] = [payer];

    let tx2 = Transaction::new_signed_with_payer(
        &[init_ix],
        Some(&payer.pubkey()),
        &signers2[..],
        recent2,
    );

    let sig2 = client.send_and_confirm_transaction(&tx2).await?;
    println!("[bridge] InitializeMarket tx: {}", sig2);

    // Store info so RESOLVE knows which account to target
    let info = MarketInfo {
        event_id,
        market_pubkey: market_pda,
        yes_mint: yes_mint_kp.pubkey(),
        no_mint: no_mint_kp.pubkey(),
        yes_vault: yes_vault_kp.pubkey(),
        no_vault: no_vault_kp.pubkey(),
    };

    MARKET_REGISTRY
        .lock()
        .unwrap()
        .insert(market_name.clone(), info);

    Ok(())
}

// --------------------------------------------------------
// RESOLVE handler: call ResolveMarket on-chain
// --------------------------------------------------------

async fn handle_resolve(
    client: &RpcClient,
    payer: &Keypair,
    program_id: &Pubkey,
    action: &Action,
) -> Result<()> {
    let market_name = &action.market_name;

    let registry = MARKET_REGISTRY.lock().unwrap();
    let info = match registry.get(market_name) {
        Some(i) => i,
        None => {
            eprintln!(
                "[bridge] Unknown market '{}' (maybe bridge restarted). Resolution skipped.",
                market_name
            );
            return Ok(());
        }
    };

    let outcome_str = action
        .outcome
        .as_deref()
        .unwrap_or("YES")
        .to_uppercase();
    let outcome: u8 = if outcome_str == "YES" { 1 } else { 2 };

    println!(
        "[bridge] Resolving market '{}' (account {}) with outcome {}",
        market_name, info.market_pubkey, outcome_str
    );

    let resolve_data =
        ProphetInstruction::ResolveMarket { outcome }.try_to_vec()?;

    let resolve_ix = Instruction {
        program_id: *program_id,
        accounts: vec![
            AccountMeta::new(info.market_pubkey, false),
            AccountMeta::new_readonly(payer.pubkey(), true),
        ],
        data: resolve_data,
    };

    let recent = client.get_latest_blockhash().await?;
    let signers: [&Keypair; 1] = [payer];

    let tx = Transaction::new_signed_with_payer(
        &[resolve_ix],
        Some(&payer.pubkey()),
        &signers[..],
        recent,
    );

    let sig = client.send_and_confirm_transaction(&tx).await?;
    println!("[bridge] ResolveMarket tx: {}", sig);

    Ok(())
}
