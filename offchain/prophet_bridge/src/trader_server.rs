// offchain/src/trader_server.rs
use crate::protocol::{ProphetInstruction, TOKEN_SCALE};
use crate::types::MARKET_REGISTRY;

use anyhow::{anyhow, Result};
use borsh::BorshSerialize;
use futures::{SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::json;
use solana_client::nonblocking::rpc_client::RpcClient;
use solana_sdk::{
    commitment_config::CommitmentConfig,
    instruction::{AccountMeta, Instruction},
    native_token::LAMPORTS_PER_SOL,
    program_pack::Pack,
    pubkey::Pubkey,
    signature::{Keypair, Signer},
    system_instruction,
    transaction::Transaction,
};
use spl_associated_token_account::{get_associated_token_address, instruction as ata_ix};
use spl_token::{id as spl_token_program_id, state::Account as TokenAccount};

use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::{Duration, SystemTime},
};
use tokio::{net::TcpListener, sync::mpsc, time::sleep};
use tokio_tungstenite::{accept_async, tungstenite::Message};
use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode;
use tokio_tungstenite::tungstenite::protocol::CloseFrame;

// ---------- Types ----------

#[derive(Clone)]
pub struct TraderServerConfig {
    pub listen_addr: SocketAddr, // e.g., "0.0.0.0:8767".parse().unwrap()
    pub program_id: Pubkey,
}

#[derive(Clone)]
struct TraderConn {
    // where to send outbound msgs
    tx: mpsc::UnboundedSender<serde_json::Value>,
    // the agent's solana keypair (custodied by the server for dev)
    agent_kp: Arc<Keypair>,
}


#[derive(Deserialize)]
struct PlaceOrderMsg {
    action: String,
    market: String,      // e.g., "MarketName_YES" or "MarketName_NO"
    side: String,        // "buy" | "sell"
    price: Option<f64>,  // optional hint for UI / logging
    quantity: f64,       // number of YES/NO shares (float)
}

// AMM-style message: signed quantity => buy (>=0) or sell (<0)
#[derive(Deserialize)]
struct TradeMsg {
    action: String,   // "trade"
    market: String,   // e.g., "MarketName_YES" or "MarketName_NO"
    quantity: f64,    // positive => buy, negative => sell
}

#[derive(Deserialize)]
#[serde(tag = "action")]
enum InboundMsg {
    #[serde(rename = "register")]
    Register { agent_id: String },
    #[serde(rename = "register_spectator")]
    RegisterSpectator { client_id: Option<String> },
    #[serde(rename = "register_manager")]
    RegisterManager { client_id: Option<String> },
    #[serde(other)]
    Other,
}


// ---------- Server ----------

pub struct TraderServer {
    cfg: TraderServerConfig,
    rpc: Arc<RpcClient>,
    payer: Arc<Keypair>, // used to fund agents
    // connected trading agents
    conns: Arc<Mutex<HashMap<String, TraderConn>>>,
    // connected spectators (frontend dashboards, etc.)
    spectators: Arc<Mutex<HashMap<String, mpsc::UnboundedSender<serde_json::Value>>>>,
}

impl TraderServer {
    pub fn new(cfg: TraderServerConfig, rpc_url: String, payer: Arc<Keypair>) -> Self {
        let rpc = Arc::new(RpcClient::new_with_commitment(
            rpc_url,
            CommitmentConfig::confirmed(),
        ));
        Self {
            cfg,
            rpc,
            payer,
            conns: Arc::new(Mutex::new(HashMap::new())),
            spectators: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Start WS server and a periodic “price ticker” broadcaster.
    pub async fn run(self: Arc<Self>) -> Result<()> {
        let listener = TcpListener::bind(self.cfg.listen_addr).await?;
        println!("[trader-server] listening on ws://{}", self.cfg.listen_addr);

        // spawn periodic broadcaster for synthetic order_book_update + price_update
        let me = self.clone();
        tokio::spawn(async move { me.periodic_price_broadcast().await });

        loop {
            let (stream, addr) = listener.accept().await?;
            let me = self.clone();
            tokio::spawn(async move {
                if let Err(e) = me.handle_conn(stream, addr).await {
                    eprintln!("[trader-server] conn error from {addr}: {e:?}");
                }
            });
        }
    }

    async fn handle_conn(
        self: Arc<Self>,
        stream: tokio::net::TcpStream,
        addr: SocketAddr,
    ) -> Result<()> {
        let ws = accept_async(stream).await?;
        println!("[trader-server] new connection from {addr}");

        // channel to send messages to this client
        let (out_tx, mut out_rx) = mpsc::unbounded_channel::<serde_json::Value>();

        // split WS
        let (mut ws_tx, mut ws_rx) = ws.split();

        // writer task
        let write_task = tokio::spawn(async move {
            while let Some(msg) = out_rx.recv().await {
                let text = serde_json::to_string(&msg).unwrap();
                if ws_tx.send(Message::Text(text)).await.is_err() {
                    break;
                }
            }
            // close
            let _ = ws_tx
                .send(Message::Close(Some(CloseFrame {
                    code: CloseCode::Normal,
                    reason: "bye".into(),
                })))
                .await;
        });

        // until registered we keep a temporary sender
        let mut agent_or_spec_id: Option<String> = None;
        let mut is_spectator = false;

        // reader loop
        while let Some(Ok(msg)) = ws_rx.next().await {
            if !msg.is_text() {
                continue;
            }
            let text = msg.into_text()?;

            // Try to parse as register / register_spectator first
            if agent_or_spec_id.is_none() {
                match serde_json::from_str::<InboundMsg>(&text) {
                    Ok(InboundMsg::Register { agent_id: aid }) => {
                        // normal trading agent
                        let kp = self.load_or_create_agent_kp(&aid).await?;
                        self.ensure_agent_funded(&kp).await?;

                        let conn = TraderConn {
                            tx: out_tx.clone(),
                            agent_kp: Arc::new(kp),
                        };
                        self.conns.lock().unwrap().insert(aid.clone(), conn);

                        agent_or_spec_id = Some(aid.clone());
                        is_spectator = false;

                        // ack & initial balances
                        self.send_to(&aid, json!({"type":"registered","agent_id": aid}))?;
                        self.push_account_update(&aid).await?;

                        println!("[trader-server] registered agent {aid}");
                    }
                    Ok(InboundMsg::RegisterSpectator { client_id }) => {
                        // spectator (frontend dashboard)
                        let ts = SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_millis();
                        let sid = client_id.unwrap_or_else(|| format!("spectator_{ts}"));

                        {
                            let mut specs = self.spectators.lock().unwrap();
                            specs.insert(sid.clone(), out_tx.clone());
                        }

                        agent_or_spec_id = Some(sid.clone());
                        is_spectator = true;

                        // ack
                        let _ = out_tx.send(json!({
                            "type": "spectator_registered",
                            "spectator_id": sid
                        }));

                        // Send snapshot of all known accounts + prices
                        if let Err(e) = self.push_all_accounts_snapshot_to(&out_tx).await {
                            eprintln!("[trader-server] failed to push account snapshot to spectator: {e:?}");
                        }
                        if let Err(e) = self.snapshot_prices_to(&out_tx).await {
                            eprintln!("[trader-server] failed to push price snapshot to spectator: {e:?}");
                        }

                        println!("[trader-server] registered spectator {sid}");
                    }
					Ok(InboundMsg::RegisterManager { client_id }) => {
						let ts = SystemTime::now()
							.duration_since(std::time::UNIX_EPOCH)
							.unwrap_or_default()
							.as_millis();
						let mid = client_id.unwrap_or_else(|| format!("manager_{ts}"));

						{
							// For now, reuse spectators map so manager sees all broadcasts
							let mut specs = self.spectators.lock().unwrap();
							specs.insert(mid.clone(), out_tx.clone());
						}

						agent_or_spec_id = Some(mid.clone());
						is_spectator = true;

						let _ = out_tx.send(json!({
							"type": "manager_registered",
							"manager_id": mid
						}));

						// Optional: send initial snapshots just like for spectators
						if let Err(e) = self.push_all_accounts_snapshot_to(&out_tx).await {
							eprintln!("[trader-server] failed to push account snapshot to manager: {e:?}");
						}
						if let Err(e) = self.snapshot_prices_to(&out_tx).await {
							eprintln!("[trader-server] failed to push price snapshot to manager: {e:?}");
						}

						println!("[trader-server] registered manager {mid}");
					}
                    _ => {
                        // Require some form of registration first
                        let _ = out_tx.send(json!({
                            "type":"error",
                            "message":"You must register first"
                        }));
                    }
                }
                continue;
            }

            // From here we have an ID
            let aid = agent_or_spec_id.clone().unwrap();

            // spectators don't send trading commands (ignore anything else)
            if is_spectator {
                continue;
            }

            // 1) Try AMM-style "trade" message (quantity sign => side)
            if let Ok(tr) = serde_json::from_str::<TradeMsg>(&text) {
                if tr.action == "trade" {
                    let side = if tr.quantity >= 0.0 {
                        "buy".to_string()
                    } else {
                        "sell".to_string()
                    };
                    let po = PlaceOrderMsg {
                        action: "place_order".to_string(),
                        market: tr.market,
                        side,
                        price: None,
                        quantity: tr.quantity.abs(),
                    };

                    if let Err(e) = self.handle_place_order(&aid, po).await {
                        self.send_to(&aid, json!({"type":"error","message": format!("{e}")}))?;
                    } else {
                        // after tx, push balances (to agent + spectators)
                        self.push_account_update(&aid).await?;
                    }
                    continue;
                }
            }

            // 2) Fallback to legacy "place_order"
            if let Ok(po) = serde_json::from_str::<PlaceOrderMsg>(&text) {
                if po.action == "place_order" {
                    if let Err(e) = self.handle_place_order(&aid, po).await {
                        self.send_to(&aid, json!({"type":"error","message": format!("{e}")}))?;
                    } else {
                        // after tx, push balances (to agent + spectators)
                        self.push_account_update(&aid).await?;
                    }
                    continue;
                }
            }

            // Unknown
            self.send_to(&aid, json!({"type":"error","message":"Unknown/invalid message"}))?;
        }

        // drop
        write_task.abort();
        if let Some(id) = agent_or_spec_id {
            if is_spectator {
                self.spectators.lock().unwrap().remove(&id);
                println!("[trader-server] spectator {id} disconnected");
            } else {
                self.conns.lock().unwrap().remove(&id);
                println!("[trader-server] agent {id} disconnected");
            }
        }
        Ok(())
    }

    // ---------- Public events (called by main.rs) ----------

    pub fn broadcast_new_market(&self, market_name: &str, implied_prob_yes: Option<f64>) -> Result<()> {
        let msg = json!({
            "type":"new_market",
            "market_name": market_name,
            "market_yes": format!("{market_name}_YES"),
            "market_no":  format!("{market_name}_NO"),
            "implied_prob_yes": implied_prob_yes,
            "created_by": "onchain"
        });
        self.broadcast(msg)
    }

    pub fn broadcast_market_resolved(&self, market_name: &str, outcome: &str) -> Result<()> {
        let msg = json!({
            "type":"market_resolved",
            "market_name": market_name,
            "outcome": outcome
        });
        self.broadcast(msg)
    }

    // ---------- Helpers ----------

    fn send_to(&self, agent_id: &str, payload: serde_json::Value) -> Result<()> {
        if let Some(conn) = self.conns.lock().unwrap().get(agent_id) {
            conn.tx
                .send(payload)
                .map_err(|_| anyhow!("failed sending to agent {agent_id}"))?;
        }
        Ok(())
    }

    fn broadcast(&self, payload: serde_json::Value) -> Result<()> {
        // send to trading agents
        {
            let conns = self.conns.lock().unwrap();
            for (aid, conn) in conns.iter() {
                if conn.tx.send(payload.clone()).is_err() {
                    eprintln!("[trader-server] failed to send to agent {aid}");
                }
            }
        }
        // send to spectators
        {
            let specs = self.spectators.lock().unwrap();
            for (sid, tx) in specs.iter() {
                if tx.send(payload.clone()).is_err() {
                    eprintln!("[trader-server] failed to send to spectator {sid}");
                }
            }
        }
        Ok(())
    }

    async fn collect_balances_for_keypair(
        &self,
        kp: &Keypair,
    ) -> Result<serde_json::Map<String, serde_json::Value>> {
        let lamports = self.rpc.get_balance(&kp.pubkey()).await?;
        let mut balances = serde_json::Map::new();
		let ui_balance = (lamports as f64) / (LAMPORTS_PER_SOL as f64);
		balances.insert("USD".to_string(), json!(ui_balance * 10_000.0));

        // scan known markets for ATAs
        let registry = MARKET_REGISTRY.lock().unwrap().clone();
        for (mname, info) in registry {
            let yes_ata = get_associated_token_address(&kp.pubkey(), &info.yes_mint);
            if let Ok(bal) = self.try_token_balance(&yes_ata).await {
                if bal > 0.0 {
                    balances.insert(format!("{}_YES", mname), json!(bal));
                }
            }
            let no_ata = get_associated_token_address(&kp.pubkey(), &info.no_mint);
            if let Ok(bal) = self.try_token_balance(&no_ata).await {
                if bal > 0.0 {
                    balances.insert(format!("{}_NO", mname), json!(bal));
                }
            }
        }

        Ok(balances)
    }

    /// Sends a fresh account_update to:
    ///   - the trading agent itself (no agent_id field)
    ///   - all spectators (with agent_id field for scoreboard)
    async fn push_account_update(&self, agent_id: &str) -> Result<()> {
        let kp = {
            let conns = self.conns.lock().unwrap();
            let c = conns.get(agent_id).ok_or_else(|| anyhow!("not connected"))?;
            c.agent_kp.clone()
        };

        let balances = self.collect_balances_for_keypair(&kp).await?;

        // 1) send to the agent (matches your Python trader expectations)
        self.send_to(agent_id, json!({
            "type": "account_update",
            "balances": balances
        }))?;

        // 2) send to all spectators with explicit agent_id
        let msg_for_specs = json!({
            "type": "account_update",
            "agent_id": agent_id,
            "balances": balances
        });
        {
            let specs = self.spectators.lock().unwrap();
            for (sid, tx) in specs.iter() {
                if tx.send(msg_for_specs.clone()).is_err() {
                    eprintln!("[trader-server] failed to send account_update for {agent_id} to spectator {sid}");
                }
            }
        }

        Ok(())
    }

    /// Sends a snapshot of all known agents' accounts to a single spectator
    async fn push_all_accounts_snapshot_to(
        &self,
        out_tx: &mpsc::UnboundedSender<serde_json::Value>,
    ) -> Result<()> {
        let conns = self.conns.lock().unwrap().clone();

        for (agent_id, conn) in conns {
            let balances = self.collect_balances_for_keypair(&conn.agent_kp).await?;
            let _ = out_tx.send(json!({
                "type": "account_update",
                "agent_id": agent_id,
                "balances": balances
            }));
        }

        Ok(())
    }

    /// Sends a one-shot prices snapshot to a single spectator
    async fn snapshot_prices_to(
        &self,
        out_tx: &mpsc::UnboundedSender<serde_json::Value>,
    ) -> Result<()> {
        let registry = MARKET_REGISTRY.lock().unwrap().clone();

        for (mname, info) in registry {
            if let Ok(acc) = self.rpc.get_account(&info.market_pubkey).await {
                let data = acc.data;
                if data.len()
                    < 32 + 32 + 1 + 8 + 32 + 32 + 8 + 8 + 8 + 1 + 1
                {
                    continue;
                }
                // layout: authority(32) + event_id(32) + category(1) + end_ts(8)
                // yes_mint(32) + no_mint(32) + b(8) + q_yes(8) + q_no(8) + resolved(1) + outcome(1)
                let mut off = 32 + 32 + 1 + 8 + 32 + 32;
                let b = f64::from_le_bytes(data[off..off + 8].try_into().unwrap());
                off += 8;
                let qy = f64::from_le_bytes(data[off..off + 8].try_into().unwrap());
                off += 8;
                let qn = f64::from_le_bytes(data[off..off + 8].try_into().unwrap());
                off += 8;
                let resolved = data[off] != 0;
                if resolved {
                    continue;
                }

                let px_yes = price_yes(b, qy, qn);
                let px_no = 1.0 - px_yes;

                let msg = json!({
                    "type": "price_update",
                    "market": mname,
                    "price_yes": round3(px_yes),
                    "price_no":  round3(px_no),
                    "q_yes": qy,
                    "q_no":  qn,
                });

                let _ = out_tx.send(msg);
            }
        }
        Ok(())
    }

    async fn try_token_balance(&self, ata: &Pubkey) -> Result<f64> {
        let acc = self.rpc.get_account(ata).await?;
        if acc.owner != spl_token_program_id() {
            return Ok(0.0);
        }
        let tok = TokenAccount::unpack_from_slice(&acc.data)?;
        let ui = (tok.amount as f64) / (TOKEN_SCALE as f64);
        Ok(ui)
    }

    async fn handle_place_order(&self, agent_id: &str, po: PlaceOrderMsg) -> Result<()> {
        // parse market suffix
        let (base, is_yes) = if po.market.ends_with("_YES") {
            (po.market.strip_suffix("_YES").unwrap().to_string(), true)
        } else if po.market.ends_with("_NO") {
            (po.market.strip_suffix("_NO").unwrap().to_string(), false)
        } else {
            return Err(anyhow!("market must end with _YES or _NO"));
        };

        // Pull MarketInfo out while holding the lock, then drop the lock
        let info = {
            let reg = MARKET_REGISTRY.lock().unwrap();
            reg.get(&base)
                .ok_or_else(|| anyhow!("unknown market {base}"))?
                .clone()
        };

        // find agent conn + compute units
        let (agent_kp, amount_units): (Arc<Keypair>, u64) = {
            let conns = self.conns.lock().unwrap();
            let c = conns
                .get(agent_id)
                .ok_or_else(|| anyhow!("not connected"))?;
            let units = (po.quantity * (TOKEN_SCALE as f64)).round() as u64;
            (c.agent_kp.clone(), units)
        };

        if amount_units == 0 {
            return Err(anyhow!("quantity too small"));
        }

        // ensure ATAs exist for (agent, chosen mint)
        let user_ata = self
            .ensure_ata(
                &agent_kp,
                if is_yes { info.yes_mint } else { info.no_mint },
            )
            .await?;

        // Build instruction
        let data = match (po.side.as_str(), is_yes) {
            ("buy", true) => ProphetInstruction::BuyYes {
                amount: amount_units,
            }
            .try_to_vec()?,
            ("sell", true) => ProphetInstruction::SellYes {
                amount: amount_units,
            }
            .try_to_vec()?,
            ("buy", false) => ProphetInstruction::BuyNo {
                amount: amount_units,
            }
            .try_to_vec()?,
            ("sell", false) => ProphetInstruction::SellNo {
                amount: amount_units,
            }
            .try_to_vec()?,
            _ => return Err(anyhow!("side must be buy|sell")),
        };

        let mint = if is_yes { info.yes_mint } else { info.no_mint };

        let ix = Instruction {
            program_id: self.cfg.program_id,
            accounts: vec![
                AccountMeta::new(info.market_pubkey, false), // market PDA
                AccountMeta::new(agent_kp.pubkey(), true),   // user (signer)
                AccountMeta::new(user_ata, false),           // user token account
                AccountMeta::new(mint, false),               // mint
                AccountMeta::new_readonly(spl_token_program_id(), false), // token prog
                AccountMeta::new_readonly(
                    solana_sdk::system_program::id(),
                    false,
                ), // system
            ],
            data,
        };

        // Track balance delta to compute approximate "cost" of this trade in SOL
        let pre_balance = self.rpc.get_balance(&agent_kp.pubkey()).await?;

        let recent = self.rpc.get_latest_blockhash().await?;
        let tx = Transaction::new_signed_with_payer(
            &[ix],
            Some(&agent_kp.pubkey()),
            &[agent_kp.as_ref()],
            recent,
        );

        let sig = self.rpc.send_and_confirm_transaction(&tx).await?;
        println!(
            "[trader-server] {} {} {} (sig={sig})",
            agent_id, po.side, po.market
        );

        let post_balance = self.rpc.get_balance(&agent_kp.pubkey()).await?;
        let delta_lamports = post_balance as i64 - pre_balance as i64;
        // cost semantics:
        //  - BUY: cost > 0 (you spent SOL)
        //  - SELL: cost < 0 (you received SOL)
        let cost_sol = -(delta_lamports as f64) / (LAMPORTS_PER_SOL as f64);

        // Emit a “trade_executed” UX message similar to your Python AMM
        self.broadcast(json!({
            "type":"trade_executed",
            "market": po.market,
            "quantity": po.quantity,
            "price": po.price,      // optional hint
            "side": po.side,
            "trader": agent_id,     // older field
            "agent": agent_id,      // new field for bots
            "cost": cost_sol        // Δcash in SOL units
        }))?;

        Ok(())
    }

    async fn ensure_ata(&self, owner: &Keypair, mint: Pubkey) -> Result<Pubkey> {
        let ata = get_associated_token_address(&owner.pubkey(), &mint);
        if self.rpc.get_account(&ata).await.is_ok() {
            return Ok(ata);
        }

        // create associated token account (payer = owner)
        let ix = ata_ix::create_associated_token_account(
            &owner.pubkey(),
            &owner.pubkey(),
            &mint,
            &spl_token_program_id(),
        );

        let recent = self.rpc.get_latest_blockhash().await?;
        let tx =
            Transaction::new_signed_with_payer(&[ix], Some(&owner.pubkey()), &[owner], recent);
        let _sig = self.rpc.send_and_confirm_transaction(&tx).await?;
        Ok(ata)
    }

    async fn load_or_create_agent_kp(&self, agent_id: &str) -> Result<Keypair> {
        let mut path = dirs::home_dir().ok_or_else(|| anyhow!("no home dir"))?;
        path.push(".config/prophet/agents");
        std::fs::create_dir_all(&path)?;
        path.push(format!("{agent_id}.json"));

        if path.exists() {
            let data = std::fs::read_to_string(&path)?;
            let bytes: Vec<u8> = serde_json::from_str(&data)?;
            return Ok(Keypair::from_bytes(&bytes)?);
        }

        let kp = Keypair::new();
        let bytes = kp.to_bytes().to_vec();
        std::fs::write(&path, serde_json::to_string(&bytes)?)?;
        Ok(kp)
    }

    async fn ensure_agent_funded(&self, agent_kp: &Keypair) -> Result<()> {
        let min = (0.05 * LAMPORTS_PER_SOL as f64) as u64; // ~0.05 SOL for fees + small trades
        let bal = self.rpc.get_balance(&agent_kp.pubkey()).await?;
        if bal >= min {
            return Ok(());
        }

        // transfer from payer to agent
        let ix =
            system_instruction::transfer(&self.payer.pubkey(), &agent_kp.pubkey(), min);
        let recent = self.rpc.get_latest_blockhash().await?;
        let tx = Transaction::new_signed_with_payer(
            &[ix],
            Some(&self.payer.pubkey()),
            &[self.payer.as_ref()],
            recent,
        );
        let _sig = self.rpc.send_and_confirm_transaction(&tx).await?;
        Ok(())
    }

    async fn periodic_price_broadcast(self: Arc<Self>) {
        loop {
            if let Err(e) = self.tick_prices().await {
                eprintln!("[trader-server] price tick error: {e:?}");
            }
            sleep(Duration::from_secs(2)).await;
        }
    }

    async fn tick_prices(&self) -> Result<()> {
        // For each known market, read account data, compute p_yes/p_no
        let registry = MARKET_REGISTRY.lock().unwrap().clone();

        for (mname, info) in registry {
            if let Ok(acc) = self.rpc.get_account(&info.market_pubkey).await {
                // layout matches on-chain Market:
                // authority(32) + event_id(32) + category(1) + end_ts(8)
                // yes_mint(32) + no_mint(32) + b(8) + q_yes(8) + q_no(8)
                // resolved(1) + outcome(1)
                let data = acc.data;
                if data.len()
                    < 32 + 32 + 1 + 8 + 32 + 32 + 8 + 8 + 8 + 1 + 1
                {
                    continue;
                }
                let mut off = 32 + 32 + 1 + 8 + 32 + 32;
                let b = f64::from_le_bytes(data[off..off + 8].try_into().unwrap());
                off += 8;
                let qy = f64::from_le_bytes(data[off..off + 8].try_into().unwrap());
                off += 8;
                let qn = f64::from_le_bytes(data[off..off + 8].try_into().unwrap());
                off += 8;
                let resolved = data[off] != 0;
                if resolved {
                    continue;
                }

                let px_yes = price_yes(b, qy, qn);
                let px_no = 1.0 - px_yes;

                // Synthetic order_book_update for older bots (your current agentic_traders)
                let msg_yes = json!({
                    "type":"order_book_update",
                    "market": format!("{}_YES", mname),
                    "bids": [[round3(px_yes - 0.01), 1_000_000.0, "AMM"]],
                    "asks": [[round3(px_yes + 0.01), 1_000_000.0, "AMM"]]
                });
                let msg_no = json!({
                    "type":"order_book_update",
                    "market": format!("{}_NO", mname),
                    "bids": [[round3(px_no - 0.01), 1_000_000.0, "AMM"]],
                    "asks": [[round3(px_no + 0.01), 1_000_000.0, "AMM"]]
                });

                self.broadcast(msg_yes)?;
                self.broadcast(msg_no)?;

                // Clean LMSR price_update for AMM-style bots + frontend
                let msg_price = json!({
                    "type":"price_update",
                    "market": mname,
                    "price_yes": round3(px_yes),
                    "price_no":  round3(px_no),
                    "q_yes": qy,
                    "q_no":  qn,
                });

                self.broadcast(msg_price)?;
            }
        }
        Ok(())
    }
}

// ---------- small helpers ----------

fn price_yes(b: f64, qy: f64, qn: f64) -> f64 {
    // p_yes = e^(qy/b) / (e^(qy/b) + e^(qn/b))
    let x = (qy / b).exp();
    let y = (qn / b).exp();
    x / (x + y)
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}
