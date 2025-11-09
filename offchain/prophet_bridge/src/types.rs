// offchain/src/types.rs
use once_cell::sync::Lazy;
use solana_sdk::pubkey::Pubkey;
use std::collections::HashMap;
use std::sync::Mutex;

#[derive(Clone, Debug)]
pub struct MarketInfo {
    pub event_id: [u8; 32],
    pub market_pubkey: Pubkey,
    pub yes_mint: Pubkey,
    pub no_mint: Pubkey,
}

// Global in-memory registry so both main.rs and the trader server see the same markets
pub static MARKET_REGISTRY: Lazy<Mutex<HashMap<String, MarketInfo>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
