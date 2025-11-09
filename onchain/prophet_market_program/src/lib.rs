use borsh::{BorshDeserialize, BorshSerialize};
use libm::{exp, log};
use solana_program::{
    account_info::{next_account_info, AccountInfo},
    entrypoint,
    entrypoint::ProgramResult,
    msg,
    program::{invoke, invoke_signed},
    program_error::ProgramError,
    pubkey::Pubkey,
    system_instruction,
    system_program,
    sysvar::{rent::Rent, Sysvar},
};

use spl_token::instruction as token_instruction;

// Seed used to derive the Market PDA
pub const MARKET_SEED: &[u8] = b"market";

// How big the Market account data should be.
// 256 is a nice round, safe value (> Borsh size of Market).
pub const MARKET_ACCOUNT_SIZE: usize = 256;

// Shares are measured in units corresponding to the token's smallest unit.
// If your YES/NO mints have 6 decimals, 1 share = 10^6 units.
const SHARE_SCALE: f64 = 1_000_000.0;

// LMSR liquidity parameter. Larger b = deeper market / less slippage.
const DEFAULT_B: f64 = 10_000.0;

/// On-chain state for a single prediction market using LMSR.
/// - Collateral is native SOL (lamports held in the Market PDA).
/// - YES/NO are SPL tokens minted/burned by the PDA.
/// - q_yes / q_no are *share counts* (in whole shares, not raw token units).
#[derive(BorshSerialize, BorshDeserialize, Debug, Clone)]
pub struct Market {
    pub authority: Pubkey,      // who can resolve this market
    pub event_id: [u8; 32],     // unique ID for the event
    pub category: u8,           // arbitrary category code
    pub end_timestamp: i64,     // unix timestamp of market end

    pub yes_mint: Pubkey,       // SPL mint for YES shares
    pub no_mint: Pubkey,        // SPL mint for NO shares

    pub b: f64,                 // LMSR liquidity parameter
    pub q_yes: f64,             // total YES shares outstanding
    pub q_no: f64,              // total NO shares outstanding

    pub resolved: bool,         // has this market been resolved?
    pub outcome: u8,            // 0 = unknown, 1 = YES, 2 = NO
}

/// Instructions supported by the Prophet market program.
///
/// NOTE: the *data* layout of this enum must match your off-chain code.
/// The account layouts have changed compared to the old fixed-price version.
#[derive(BorshSerialize, BorshDeserialize, Debug, Clone)]
pub enum ProphetInstruction {
    /// InitializeMarket {
    ///   event_id: [u8; 32],
    ///   category: u8,
    ///   end_timestamp: i64,
    /// }
    ///
    /// Accounts:
    /// 0. [writable] Market PDA account (PDA derived from [b"market", event_id])
    /// 1. [signer]   Authority (payer, who can later resolve the market)
    /// 2. []         YES mint (SPL mint, mint_authority = Market PDA)
    /// 3. []         NO mint  (SPL mint, mint_authority = Market PDA)
    /// 4. []         Token program (SPL Token)
    /// 5. []         System program
    InitializeMarket {
        event_id: [u8; 32],
        category: u8,
        end_timestamp: i64,
    },

    /// BuyYes { amount: u64 }
    ///
    /// amount = number of YES *token units* to receive (e.g. with 6 decimals,
    /// amount = shares * 10^6).
    ///
    /// The AMM calculates the LMSR cost in SOL and charges that many lamports
    /// from the user, minting `amount` YES tokens to them.
    ///
    /// Accounts:
    /// 0. [writable] Market PDA account
    /// 1. [signer, writable] User (payer, SOL source)
    /// 2. [writable] User YES token account (ATA)
    /// 3. [writable] YES mint
    /// 4. []         Token program
    /// 5. []         System program
    BuyYes {
        amount: u64,
    },

    /// SellYes { amount: u64 }
    ///
    /// amount = number of YES *token units* the user sells back to the AMM.
    /// The AMM computes the LMSR refund in SOL and transfers that many
    /// lamports from the Market PDA to the user, burning the YES tokens.
    ///
    /// Accounts:
    /// 0. [writable] Market PDA account
    /// 1. [signer, writable] User
    /// 2. [writable] User YES token account
    /// 3. [writable] YES mint
    /// 4. []         Token program
    /// 5. []         System program
    SellYes {
        amount: u64,
    },

    /// BuyNo { amount: u64 }
    ///
    /// Same semantics as BuyYes, but for NO shares.
    ///
    /// Accounts:
    /// 0. [writable] Market PDA account
    /// 1. [signer, writable] User
    /// 2. [writable] User NO token account
    /// 3. [writable] NO mint
    /// 4. []         Token program
    /// 5. []         System program
    BuyNo {
        amount: u64,
    },

    /// SellNo { amount: u64 }
    ///
    /// Same semantics as SellYes, but for NO shares.
    ///
    /// Accounts:
    /// 0. [writable] Market PDA account
    /// 1. [signer, writable] User
    /// 2. [writable] User NO token account
    /// 3. [writable] NO mint
    /// 4. []         Token program
    /// 5. []         System program
    SellNo {
        amount: u64,
    },

    /// ResolveMarket { outcome: u8 }
    ///
    /// outcome: 1 = YES, 2 = NO
    ///
    /// Accounts:
    /// 0. [writable] Market PDA account
    /// 1. [signer]   Authority (must match Market.authority)
    ResolveMarket {
        outcome: u8,
    },
}

entrypoint!(process_instruction);

/// Program entrypoint: dispatch based on serialized ProphetInstruction.
pub fn process_instruction(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    instruction_data: &[u8],
) -> ProgramResult {
    let ix = ProphetInstruction::try_from_slice(instruction_data)
        .map_err(|_| ProgramError::InvalidInstructionData)?;

    match ix {
        ProphetInstruction::InitializeMarket {
            event_id,
            category,
            end_timestamp,
        } => process_initialize_market(program_id, accounts, event_id, category, end_timestamp),
        ProphetInstruction::BuyYes { amount } => process_buy(
            program_id,
            accounts,
            amount,
            true,  // is_yes
            true,  // is_buy
        ),
        ProphetInstruction::SellYes { amount } => process_buy(
            program_id,
            accounts,
            amount,
            true,  // is_yes
            false, // is_buy
        ),
        ProphetInstruction::BuyNo { amount } => process_buy(
            program_id,
            accounts,
            amount,
            false, // is_yes
            true,  // is_buy
        ),
        ProphetInstruction::SellNo { amount } => process_buy(
            program_id,
            accounts,
            amount,
            false, // is_yes
            false, // is_buy
        ),
        ProphetInstruction::ResolveMarket { outcome } => {
            process_resolve_market(program_id, accounts, outcome)
        }
    }
}

/// Helper: deserialize Market from account data.
fn load_market(market_ai: &AccountInfo) -> Result<Market, ProgramError> {
    let data = market_ai.try_borrow_data()?;
    let mut slice: &[u8] = &data; // Borsh will advance this as it deserializes
    Market::deserialize(&mut slice).map_err(|_| ProgramError::InvalidAccountData)
}

/// Helper: serialize Market back into account data.
fn store_market(market_ai: &AccountInfo, market: &Market) -> Result<(), ProgramError> {
    let mut data = market_ai.try_borrow_mut_data()?;
    market
        .serialize(&mut *data)
        .map_err(|_| ProgramError::AccountDataTooSmall)
}

/// LMSR cost function:
/// C(q_yes, q_no) = b * log(exp(q_yes/b) + exp(q_no/b))
fn lmsr_cost(b: f64, q_yes: f64, q_no: f64) -> f64 {
    // log-sum-exp for numerical stability
    let x = q_yes / b;
    let y = q_no / b;
    let m = if x > y { x } else { y };
    let sum = exp(x - m) + exp(y - m);
    b * (m + log(sum))
}

/// Compute the cost (for buy) or refund (for sell) when trading `delta_shares`
/// of YES or NO, in terms of SOL (lamports, but returned as f64).
///
/// - side_is_yes = true  -> trading YES
/// - side_is_yes = false -> trading NO
/// - is_buy = true       -> cost = C(new) - C(old)
/// - is_buy = false      -> refund = C(old) - C(new)
fn lmsr_cost_delta(
    b: f64,
    q_yes: f64,
    q_no: f64,
    delta_shares: f64,
    side_is_yes: bool,
    is_buy: bool,
) -> Result<f64, ProgramError> {
    if delta_shares <= 0.0 {
        msg!("Delta must be positive");
        return Err(ProgramError::InvalidInstructionData);
    }

    let mut qy_old = q_yes;
    let mut qn_old = q_no;
    let mut qy_new = q_yes;
    let mut qn_new = q_no;

    if side_is_yes {
        if is_buy {
            qy_new += delta_shares;
        } else {
            if delta_shares > qy_old + 1e-9 {
                msg!("Cannot sell more YES than outstanding");
                return Err(ProgramError::InvalidInstructionData);
            }
            qy_new -= delta_shares;
        }
    } else {
        if is_buy {
            qn_new += delta_shares;
        } else {
            if delta_shares > qn_old + 1e-9 {
                msg!("Cannot sell more NO than outstanding");
                return Err(ProgramError::InvalidInstructionData);
            }
            qn_new -= delta_shares;
        }
    }

    let before = lmsr_cost(b, qy_old, qn_old);
    let after = lmsr_cost(b, qy_new, qn_new);
    let diff = if is_buy { after - before } else { before - after };

    if diff <= 0.0 {
        msg!("Non-positive LMSR cost delta");
        return Err(ProgramError::InvalidInstructionData);
    }

    Ok(diff)
}

/// Initialize a new Market in its PDA account.
///
/// Off-chain:
/// - Compute PDA = find_program_address([b"market", event_id], program_id)
/// - Create YES/NO mints with mint_authority = PDA
/// - Call InitializeMarket with:
///   [market_pda, authority, yes_mint, no_mint, token_program, system_program]
fn process_initialize_market(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    event_id: [u8; 32],
    category: u8,
    end_timestamp: i64,
) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();

    let market_ai = next_account_info(account_info_iter)?;        // [writable] PDA
    let authority_ai = next_account_info(account_info_iter)?;     // [signer]
    let yes_mint_ai = next_account_info(account_info_iter)?;      // []
    let no_mint_ai = next_account_info(account_info_iter)?;       // []
    let token_program_ai = next_account_info(account_info_iter)?; // []
    let system_program_ai = next_account_info(account_info_iter)?; // []

    if !authority_ai.is_signer {
        msg!("Authority must be a signer");
        return Err(ProgramError::MissingRequiredSignature);
    }

    if *token_program_ai.key != spl_token::id() {
        msg!("Expected SPL Token program");
        return Err(ProgramError::IncorrectProgramId);
    }

    if *system_program_ai.key != system_program::id() {
        msg!("Expected System program");
        return Err(ProgramError::IncorrectProgramId);
    }

    // Check the PDA is correct
    let (expected_pda, bump) =
        Pubkey::find_program_address(&[MARKET_SEED, &event_id], program_id);
    if *market_ai.key != expected_pda {
        msg!("Market account key does not match PDA derived from event_id");
        return Err(ProgramError::InvalidSeeds);
    }

    // If the market account is not yet owned by this program, create it now as a PDA.
    if *market_ai.owner != *program_id {
        if *market_ai.owner != system_program::id() {
            msg!("Market account has unexpected owner");
            return Err(ProgramError::IncorrectProgramId);
        }

        let rent = Rent::get()?;
        let lamports = rent.minimum_balance(MARKET_ACCOUNT_SIZE);

        let create_ix = system_instruction::create_account(
            authority_ai.key,
            market_ai.key,
            lamports,
            MARKET_ACCOUNT_SIZE as u64,
            program_id,
        );

        invoke_signed(
            &create_ix,
            &[
                authority_ai.clone(),
                market_ai.clone(),
                system_program_ai.clone(),
            ],
            &[&[MARKET_SEED, &event_id, &[bump]]],
        )?;
    }

    let market = Market {
        authority: *authority_ai.key,
        event_id,
        category,
        end_timestamp,
        yes_mint: *yes_mint_ai.key,
        no_mint: *no_mint_ai.key,
        b: DEFAULT_B,
        q_yes: 0.0,
        q_no: 0.0,
        resolved: false,
        outcome: 0,
    };

    store_market(market_ai, &market)?;
    msg!("Market initialized (LMSR)");
    Ok(())
}

/// Shared logic for Buy/Sell YES/NO.
///
/// amount = token units (e.g. 1 share = 10^6 units if mint has 6 decimals).
fn process_buy(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    amount: u64,
    side_is_yes: bool,
    is_buy: bool,
) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();

    let market_ai = next_account_info(account_info_iter)?;
    let user_ai = next_account_info(account_info_iter)?;
    let user_ata_ai = next_account_info(account_info_iter)?;
    let mint_ai = next_account_info(account_info_iter)?;
    let token_program_ai = next_account_info(account_info_iter)?;
    let system_program_ai = next_account_info(account_info_iter)?;

    if !user_ai.is_signer {
        msg!("User must sign to trade");
        return Err(ProgramError::MissingRequiredSignature);
    }

    if *market_ai.owner != *program_id {
        msg!("Market account must be owned by this program");
        return Err(ProgramError::IncorrectProgramId);
    }

    if *token_program_ai.key != spl_token::id() {
        msg!("Expected SPL Token program");
        return Err(ProgramError::IncorrectProgramId);
    }

    if *system_program_ai.key != system_program::id() {
        msg!("Expected System program");
        return Err(ProgramError::IncorrectProgramId);
    }

    // Load market state
    let mut market_state = load_market(market_ai)?;
    if market_state.resolved {
        msg!("Market already resolved");
        return Err(ProgramError::InvalidAccountData);
    }

    // Check mint matches YES / NO as expected
    let expected_mint = if side_is_yes {
        market_state.yes_mint
    } else {
        market_state.no_mint
    };

    if *mint_ai.key != expected_mint {
        msg!("Mint account mismatch");
        return Err(ProgramError::InvalidAccountData);
    }

    if amount == 0 {
        msg!("Amount must be > 0");
        return Err(ProgramError::InvalidInstructionData);
    }

    // Convert token units -> shares (float)
    let delta_shares = (amount as f64) / SHARE_SCALE;

    // Compute LMSR cost or refund in lamports (as f64)
    let delta_cost_f = lmsr_cost_delta(
        market_state.b,
        market_state.q_yes,
        market_state.q_no,
        delta_shares,
        side_is_yes,
        is_buy,
    )?;

    let delta_cost = delta_cost_f.ceil() as u64;

    // PDA for signing SOL transfers and mint_to
    let (pda, bump) =
        Pubkey::find_program_address(&[MARKET_SEED, &market_state.event_id], program_id);
    if pda != *market_ai.key {
        msg!("PDA mismatch while trading");
        return Err(ProgramError::InvalidSeeds);
    }

    if is_buy {
        // BUY: user pays SOL -> market, receives YES/NO tokens.

        // 1) Transfer SOL from User -> Market
        let transfer_sol_ix =
            system_instruction::transfer(user_ai.key, market_ai.key, delta_cost);
        invoke(
            &transfer_sol_ix,
            &[
                user_ai.clone(),
                market_ai.clone(),
                system_program_ai.clone(),
            ],
        )?;

        // 2) Mint YES/NO tokens to user (PDA is mint authority)
        let mint_to_ix = token_instruction::mint_to(
            token_program_ai.key,
            mint_ai.key,
            user_ata_ai.key,
            market_ai.key, // mint authority = PDA
            &[],
            amount,
        )?;

        invoke_signed(
            &mint_to_ix,
            &[
                mint_ai.clone(),
                user_ata_ai.clone(),
                market_ai.clone(),
                token_program_ai.clone(),
            ],
            &[&[MARKET_SEED, &market_state.event_id, &[bump]]],
        )?;
    } else {
        // SELL: user burns YES/NO tokens, receives SOL refund.

        // 1) Burn tokens from user account (user is authority)
        let burn_ix = token_instruction::burn(
            token_program_ai.key,
            user_ata_ai.key,
            mint_ai.key,
            user_ai.key, // burn authority = user
            &[],
            amount,
        )?;

        invoke(
            &burn_ix,
            &[
                user_ata_ai.clone(),
                mint_ai.clone(),
                user_ai.clone(),
                token_program_ai.clone(),
            ],
        )?;

        // Ensure Market has enough SOL
        let market_lamports = **market_ai.lamports.borrow();
        if market_lamports < delta_cost {
            msg!("Market has insufficient SOL liquidity");
            return Err(ProgramError::InsufficientFunds);
        }

        // 2) Transfer SOL from Market -> User (PDA signs)
        let transfer_sol_ix =
            system_instruction::transfer(market_ai.key, user_ai.key, delta_cost);
        invoke_signed(
            &transfer_sol_ix,
            &[
                market_ai.clone(),
                user_ai.clone(),
                system_program_ai.clone(),
            ],
            &[&[MARKET_SEED, &market_state.event_id, &[bump]]],
        )?;
    }

    // Update LMSR share counts
    if side_is_yes {
        if is_buy {
            market_state.q_yes += delta_shares;
        } else {
            market_state.q_yes -= delta_shares;
        }
    } else {
        if is_buy {
            market_state.q_no += delta_shares;
        } else {
            market_state.q_no -= delta_shares;
        }
    }

    store_market(market_ai, &market_state)?;

    if side_is_yes && is_buy {
        msg!(
            "User bought {} YES for {} lamports (LMSR)",
            amount,
            delta_cost
        );
    } else if side_is_yes && !is_buy {
        msg!(
            "User sold {} YES for {} lamports (LMSR)",
            amount,
            delta_cost
        );
    } else if !side_is_yes && is_buy {
        msg!(
            "User bought {} NO for {} lamports (LMSR)",
            amount,
            delta_cost
        );
    } else {
        msg!(
            "User sold {} NO for {} lamports (LMSR)",
            amount,
            delta_cost
        );
    }

    Ok(())
}

/// Resolve the market: sets resolved flag and outcome (1 = YES, 2 = NO).
/// (Redemption of YES/NO into collateral will be added in a later phase.)
fn process_resolve_market(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    outcome: u8,
) -> ProgramResult {
    if outcome != 1 && outcome != 2 {
        msg!("Invalid outcome");
        return Err(ProgramError::InvalidInstructionData);
    }

    let account_info_iter = &mut accounts.iter();

    let market_ai = next_account_info(account_info_iter)?;
    let authority_ai = next_account_info(account_info_iter)?;

    if !authority_ai.is_signer {
        msg!("Authority must sign to resolve");
        return Err(ProgramError::MissingRequiredSignature);
    }

    if *market_ai.owner != *program_id {
        msg!("Market account must be owned by this program");
        return Err(ProgramError::IncorrectProgramId);
    }

    let mut market_state = load_market(market_ai)?;
    if market_state.resolved {
        msg!("Market already resolved");
        return Err(ProgramError::InvalidAccountData);
    }

    if market_state.authority != *authority_ai.key {
        msg!("Authority mismatch");
        return Err(ProgramError::IllegalOwner);
    }

    market_state.resolved = true;
    market_state.outcome = outcome;
    store_market(market_ai, &market_state)?;

    msg!("Market resolved with outcome {}", outcome);
    Ok(())
}
