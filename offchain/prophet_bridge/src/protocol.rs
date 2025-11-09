// offchain/src/protocol.rs
use borsh::{BorshDeserialize, BorshSerialize};

pub const MARKET_SEED: &[u8] = b"market";
pub const TOKEN_SCALE: u64 = 1_000_000; // must match on-chain SHARE_SCALE (6 decimals)

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone)]
pub enum ProphetInstruction {
    InitializeMarket {
        event_id: [u8; 32],
        category: u8,
        end_timestamp: i64,
    },
    BuyYes { amount: u64 },
    SellYes { amount: u64 },
    BuyNo { amount: u64 },
    SellNo { amount: u64 },
    ResolveMarket { outcome: u8 },
}
