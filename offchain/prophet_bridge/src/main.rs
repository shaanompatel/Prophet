mod protocol;
mod trader_server;
mod types;

use std::{env, fs, net::SocketAddr, path::Path, sync::Arc};
use tokio::time::{sleep, Duration};
use anyhow::Result;
use dotenvy::dotenv;
use futures::{SinkExt, StreamExt};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio_tungstenite::connect_async;
use tungstenite::Message;

use borsh::BorshSerialize;
use solana_client::nonblocking::rpc_client::RpcClient;
use solana_sdk::{
    commitment_config::CommitmentConfig,
    instruction::{AccountMeta, Instruction},
    pubkey::Pubkey,
    signature::{Keypair, Signer},
    system_instruction,
    transaction::Transaction,
};
use solana_sdk::program_pack::Pack;
use spl_token::{
    id as spl_token_program_id,
    instruction as token_instruction,
    state::Mint,
};

use crate::protocol::{ProphetInstruction, MARKET_SEED};
use crate::trader_server::{TraderServer, TraderServerConfig};
use crate::types::{MarketInfo, MARKET_REGISTRY};

const ACTION_WS_URL: &str = "ws://localhost:8766";

/// One AI action from decide_markets.py
#[derive(Debug, Deserialize)]
struct Action {
    action: String,
    market_name: String,
    probability: Option<f64>,
    outcome: Option<String>,
    reason: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();

    // --- Load env config ---
    let program_id: Pubkey = env::var("PROPHET_PROGRAM_ID")?
        .parse()
        .expect("Invalid PROPHET_PROGRAM_ID");

    let rpc_url = env::var("SOLANA_RPC_URL")
        .unwrap_or_else(|_| "https://api.devnet.solana.com".to_string());

    // Where Python trader agents connect (defaults to 127.0.0.1:8767)
    let listen_addr: SocketAddr = env::var("TRADER_SERVER_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:8767".to_string())
        .parse()
        .expect("Invalid TRADER_SERVER_ADDR");

    // --- Load payer keypair ---
    let keypair_path = dirs::home_dir()
        .expect("no home dir")
        .join(".config/solana/id.json");
    let payer = Arc::new(read_keypair(&keypair_path)?);

    // RPC client used by the bridge for CREATE / RESOLVE
    let client =
        RpcClient::new_with_commitment(rpc_url.clone(), CommitmentConfig::confirmed());

    println!("[bridge] Wallet:    {}", payer.pubkey());
    println!("[bridge] Program:   {}", program_id);
    println!("[bridge] RPC:       {}", client.url());
    println!("[bridge] Trader WS: ws://{}", listen_addr);

    // --- Start Rust trader server (Python LLM agents connect here) ---
    let trader_cfg = TraderServerConfig {
        listen_addr,
        program_id,
    };
    let trader_server = Arc::new(TraderServer::new(trader_cfg, rpc_url.clone(), payer.clone()));
    {
        let srv = trader_server.clone();
        tokio::spawn(async move {
            if let Err(e) = srv.run().await {
                eprintln!("[trader-server] fatal error: {e:?}");
            }
        });
    }

    // --- Connect to AI Action WebSocket (CREATE / RESOLVE) in a retry loop ---
    loop {
        println!("[bridge] Connecting to AI action server at {}", ACTION_WS_URL);

        match connect_async(ACTION_WS_URL).await {
            Ok((ws_stream, _)) => {
                println!("[bridge] Connected to AI action server");

                let (mut ws_write, mut ws_read) = ws_stream.split();

                while let Some(msg) = ws_read.next().await {
                    let msg = match msg {
                        Ok(m) => m,
                        Err(e) => {
                            eprintln!("[bridge] WebSocket read error: {e}. Reconnecting...");
                            break;
                        }
                    };

                    if !msg.is_text() {
                        continue;
                    }
                    let text = match msg.to_text() {
                        Ok(t) => t,
                        Err(e) => {
                            eprintln!("[bridge] Failed to read text from WS message: {e}");
                            continue;
                        }
                    };

                    match serde_json::from_str::<Action>(text) {
                        Ok(action) => {
                            println!("\n[bridge] Received Action: {:?}", action);

                            match action.action.as_str() {
                                "CREATE" => {
                                    if let Err(e) = handle_create(
                                        &client,
                                        payer.as_ref(),
                                        &program_id,
                                        &trader_server,
                                        &action,
                                    ).await {
                                        eprintln!("[bridge] Error in CREATE handler: {e:?}");
                                    } else {
                                        let _ = ws_write
                                            .send(Message::Text(r#"{"status":"ok","type":"CREATE"}"#.into()))
                                            .await;
                                    }
                                }
                                "RESOLVE" => {
                                    if let Err(e) = handle_resolve(
                                        &client,
                                        payer.as_ref(),
                                        &program_id,
                                        &trader_server,
                                        &action,
                                    ).await {
                                        eprintln!("[bridge] Error in RESOLVE handler: {e:?}");
                                    } else {
                                        let _ = ws_write
                                            .send(Message::Text(r#"{"status":"ok","type":"RESOLVE"}"#.into()))
                                            .await;
                                    }
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

                // If we drop out of the inner while-loop, the WS closed/error’d.
                eprintln!("[bridge] Action WS connection closed. Reconnecting in 5s...");
            }

            Err(e) => {
                eprintln!(
                    "[bridge] Failed to connect to AI action server {}: {e}. Retrying in 5s...",
                    ACTION_WS_URL
                );
            }
        }

        sleep(Duration::from_secs(5)).await;
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
// CREATE handler: on-chain setup + InitializeMarket + notify traders
// --------------------------------------------------------

async fn handle_create(
    client: &RpcClient,
    payer: &Keypair,
    program_id: &Pubkey,
    trader_srv: &Arc<TraderServer>,
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

    // New accounts for YES/NO mints
    let yes_mint_kp = Keypair::new();
    let no_mint_kp = Keypair::new();

    // Rent-exempt balances for SPL mints
    let mint_rent = client
        .get_minimum_balance_for_rent_exemption(Mint::LEN)
        .await?;

    // ---------- Transaction 1: create + init mints ----------

    let mut ixs = vec![];

    // YES mint account
    ixs.push(system_instruction::create_account(
        &payer.pubkey(),
        &yes_mint_kp.pubkey(),
        mint_rent,
        Mint::LEN as u64,
        &spl_token_program_id(),
    ));

    // NO mint account
    ixs.push(system_instruction::create_account(
        &payer.pubkey(),
        &no_mint_kp.pubkey(),
        mint_rent,
        Mint::LEN as u64,
        &spl_token_program_id(),
    ));

    // Initialize YES mint (6 decimals), mint_authority = Market PDA
    ixs.push(token_instruction::initialize_mint(
        &spl_token_program_id(),
        &yes_mint_kp.pubkey(),
        &market_pda, // mint authority = PDA
        None,
        6,
    )?);

    // Initialize NO mint (6 decimals), mint_authority = Market PDA
    ixs.push(token_instruction::initialize_mint(
        &spl_token_program_id(),
        &no_mint_kp.pubkey(),
        &market_pda, // mint authority = PDA
        None,
        6,
    )?);

    let recent = client.get_latest_blockhash().await?;
    let signers: [&Keypair; 3] = [payer, &yes_mint_kp, &no_mint_kp];

    let tx = Transaction::new_signed_with_payer(
        &ixs,
        Some(&payer.pubkey()),
        &signers[..],
        recent,
    );

    let sig = client.send_and_confirm_transaction(&tx).await?;
    println!("[bridge] Mint setup tx: {}", sig);

    // ---------- Transaction 2: call InitializeMarket on your program ----------

    let category: u8 = 0;       // e.g. 0 = generic
    let end_timestamp: i64 = 0; // set real expiry later if you want

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

    // Store info so RESOLVE + trader server know where the market is
    let info = MarketInfo {
        event_id,
        market_pubkey: market_pda,
        yes_mint: yes_mint_kp.pubkey(),
        no_mint: no_mint_kp.pubkey(),
    };

    MARKET_REGISTRY
        .lock()
        .unwrap()
        .insert(market_name.clone(), info.clone());

    // Tell connected trader agents about the new market
    trader_srv.broadcast_new_market(market_name, action.probability)?;

    Ok(())
}

// --------------------------------------------------------
// RESOLVE handler: call ResolveMarket on-chain + notify traders
// --------------------------------------------------------

async fn handle_resolve(
    client: &RpcClient,
    payer: &Keypair,
    program_id: &Pubkey,
    trader_srv: &Arc<TraderServer>,
    action: &Action,
) -> Result<()> {
    let market_name = &action.market_name;

    let registry = MARKET_REGISTRY.lock().unwrap();
    let info = match registry.get(market_name) {
        Some(i) => i.clone(),
        None => {
            eprintln!(
                "[bridge] Unknown market '{}' (maybe bridge restarted). Resolution skipped.",
                market_name
            );
            return Ok(());
        }
    };
    drop(registry);

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

    // Notify trader agents so they stop trading this market
    trader_srv.broadcast_market_resolved(market_name, &outcome_str)?;

    Ok(())
}
