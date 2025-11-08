use borsh::{BorshDeserialize, BorshSerialize};
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

// How big the Market account data should be
// Needs to be >= Borsh-serialized Market size (~203 bytes)
// 256 is a nice round, safe value.
pub const MARKET_ACCOUNT_SIZE: usize = 256;

/// On-chain state for a single prediction market.
#[derive(BorshSerialize, BorshDeserialize, Debug, Clone)]
pub struct Market {
    pub authority: Pubkey,      // who can resolve this market
    pub event_id: [u8; 32],     // unique ID for the event
    pub category: u8,           // arbitrary category code
    pub end_timestamp: i64,     // unix timestamp of market end
    pub yes_mint: Pubkey,       // SPL mint for YES token
    pub no_mint: Pubkey,        // SPL mint for NO token
    pub yes_vault: Pubkey,      // token account holding YES liquidity, owned by Market PDA
    pub no_vault: Pubkey,       // token account holding NO liquidity, owned by Market PDA
    pub resolved: bool,         // has this market been resolved?
    pub outcome: u8,            // 0 = unknown, 1 = YES, 2 = NO
}

/// Instructions supported by the Prophet market program.
/// This is what your off-chain code will Borsh-serialize into `instruction_data`.
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
    /// 2. []         YES mint (SPL mint)
    /// 3. []         NO mint
    /// 4. []         YES vault token account (mint = YES mint, owner = Market PDA)
    /// 5. []         NO vault token account (mint = NO mint, owner = Market PDA)
    /// 6. []         Token program (SPL Token)
    /// 7. []         System program
    InitializeMarket {
        event_id: [u8; 32],
        category: u8,
        end_timestamp: i64,
    },

    /// BuyYes { amount: u64 }
    ///
    /// Fixed-rate: 1 lamport of SOL <-> 1 smallest unit of YES token.
    ///
    /// Accounts:
    /// 0. [writable] Market PDA account
    /// 1. [signer,writable] User (payer, SOL source)
    /// 2. [writable] User YES token account (ATA)
    /// 3. [writable] YES vault token account
    /// 4. []         Token program
    /// 5. []         System program
    BuyYes {
        amount: u64,
    },

    /// SellYes { amount: u64 }
    ///
    /// User returns YES tokens, receives SOL from Market.
    ///
    /// Accounts:
    /// 0. [writable] Market PDA account
    /// 1. [signer,writable] User
    /// 2. [writable] User YES token account
    /// 3. [writable] YES vault token account
    /// 4. []         Token program
    /// 5. []         System program
    SellYes {
        amount: u64,
    },

    /// BuyNo { amount: u64 }
    ///
    /// Accounts:
    /// 0. [writable] Market PDA account
    /// 1. [signer,writable] User
    /// 2. [writable] User NO token account
    /// 3. [writable] NO vault token account
    /// 4. []         Token program
    /// 5. []         System program
    BuyNo {
        amount: u64,
    },

    /// SellNo { amount: u64 }
    ///
    /// Accounts:
    /// 0. [writable] Market PDA account
    /// 1. [signer,writable] User
    /// 2. [writable] User NO token account
    /// 3. [writable] NO vault token account
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
        ProphetInstruction::BuyYes { amount } => process_buy_yes(program_id, accounts, amount),
        ProphetInstruction::SellYes { amount } => process_sell_yes(program_id, accounts, amount),
        ProphetInstruction::BuyNo { amount } => process_buy_no(program_id, accounts, amount),
        ProphetInstruction::SellNo { amount } => process_sell_no(program_id, accounts, amount),
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

/// Initialize a new Market in its PDA account.
///
/// Off-chain:
/// - Compute PDA = find_program_address([b"market", event_id], program_id)
/// - Create YES/NO mints
/// - Create YES/NO vault token accounts with owner = PDA
/// - Call InitializeMarket with:
///   [market_pda, authority, yes_mint, no_mint, yes_vault, no_vault, token_program, system_program]
fn process_initialize_market(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    event_id: [u8; 32],
    category: u8,
    end_timestamp: i64,
) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();

    let market_ai = next_account_info(account_info_iter)?;       // [writable] PDA
    let authority_ai = next_account_info(account_info_iter)?;    // [signer]
    let yes_mint_ai = next_account_info(account_info_iter)?;     // []
    let no_mint_ai = next_account_info(account_info_iter)?;      // []
    let yes_vault_ai = next_account_info(account_info_iter)?;    // []
    let no_vault_ai = next_account_info(account_info_iter)?;     // []
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
        yes_vault: *yes_vault_ai.key,
        no_vault: *no_vault_ai.key,
        resolved: false,
        outcome: 0,
    };

    store_market(market_ai, &market)?;
    msg!("Market initialized");
    Ok(())
}

/// Buy YES tokens by sending SOL to the Market.
///
/// 1 lamport of SOL <-> 1 smallest unit of YES.
///
/// Accounts:
/// 0. [writable] Market PDA
/// 1. [signer, writable] User
/// 2. [writable] User YES token account
/// 3. [writable] YES vault token account
/// 4. []         Token program
/// 5. []         System program
fn process_buy_yes(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    amount: u64,
) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();

    let market_ai = next_account_info(account_info_iter)?;
    let user_ai = next_account_info(account_info_iter)?;
    let user_yes_ata_ai = next_account_info(account_info_iter)?;
    let yes_vault_ai = next_account_info(account_info_iter)?;
    let token_program_ai = next_account_info(account_info_iter)?;
    let system_program_ai = next_account_info(account_info_iter)?;

    if !user_ai.is_signer {
        msg!("User must sign to buy YES");
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
    let market_state = load_market(market_ai)?;
    if market_state.resolved {
        msg!("Market already resolved");
        return Err(ProgramError::InvalidAccountData);
    }
    if *yes_vault_ai.key != market_state.yes_vault {
        msg!("YES vault account mismatch");
        return Err(ProgramError::InvalidAccountData);
    }

    // 1) Transfer SOL from User -> Market
    let transfer_sol_ix =
        system_instruction::transfer(user_ai.key, market_ai.key, amount);
    invoke(
        &transfer_sol_ix,
        &[
            user_ai.clone(),
            market_ai.clone(),
            system_program_ai.clone(),
        ],
    )?;

    // 2) Transfer YES from vault -> user (PDA signs)
    let (pda, bump) =
        Pubkey::find_program_address(&[MARKET_SEED, &market_state.event_id], program_id);
    if pda != *market_ai.key {
        msg!("PDA mismatch while buying YES");
        return Err(ProgramError::InvalidSeeds);
    }

    let transfer_token_ix = token_instruction::transfer(
        token_program_ai.key,
        yes_vault_ai.key,
        user_yes_ata_ai.key,
        market_ai.key, // authority is Market PDA
        &[],
        amount,
    )?;

    invoke_signed(
        &transfer_token_ix,
        &[
            yes_vault_ai.clone(),
            user_yes_ata_ai.clone(),
            market_ai.clone(),
            token_program_ai.clone(),
        ],
        &[&[MARKET_SEED, &market_state.event_id, &[bump]]],
    )?;

    msg!("User bought {} YES", amount);
    Ok(())
}

/// Sell YES tokens: user sends YES to vault, receives SOL from Market.
fn process_sell_yes(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    amount: u64,
) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();

    let market_ai = next_account_info(account_info_iter)?;
    let user_ai = next_account_info(account_info_iter)?;
    let user_yes_ata_ai = next_account_info(account_info_iter)?;
    let yes_vault_ai = next_account_info(account_info_iter)?;
    let token_program_ai = next_account_info(account_info_iter)?;
    let system_program_ai = next_account_info(account_info_iter)?;

    if !user_ai.is_signer {
        msg!("User must sign to sell YES");
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

    let market_state = load_market(market_ai)?;
    if market_state.resolved {
        msg!("Market already resolved");
        return Err(ProgramError::InvalidAccountData);
    }
    if *yes_vault_ai.key != market_state.yes_vault {
        msg!("YES vault mismatch");
        return Err(ProgramError::InvalidAccountData);
    }

    // Ensure Market has enough SOL
    let market_lamports = **market_ai.lamports.borrow();
    if market_lamports < amount {
        msg!("Market has insufficient SOL liquidity");
        return Err(ProgramError::InsufficientFunds);
    }

    // 1) Transfer YES from user -> vault
    let transfer_token_ix = token_instruction::transfer(
        token_program_ai.key,
        user_yes_ata_ai.key,
        yes_vault_ai.key,
        user_ai.key, // authority = user
        &[],
        amount,
    )?;
    invoke(
        &transfer_token_ix,
        &[
            user_yes_ata_ai.clone(),
            yes_vault_ai.clone(),
            user_ai.clone(),
            token_program_ai.clone(),
        ],
    )?;

    // 2) Transfer SOL from Market -> User (PDA signs)
    let (pda, bump) =
        Pubkey::find_program_address(&[MARKET_SEED, &market_state.event_id], program_id);
    if pda != *market_ai.key {
        msg!("PDA mismatch while selling YES");
        return Err(ProgramError::InvalidSeeds);
    }

    let transfer_sol_ix =
        system_instruction::transfer(market_ai.key, user_ai.key, amount);
    invoke_signed(
        &transfer_sol_ix,
        &[
            market_ai.clone(),
            user_ai.clone(),
            system_program_ai.clone(),
        ],
        &[&[MARKET_SEED, &market_state.event_id, &[bump]]],
    )?;

    msg!("User sold {} YES", amount);
    Ok(())
}

/// Buy NO tokens by sending SOL to the Market.
fn process_buy_no(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    amount: u64,
) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();

    let market_ai = next_account_info(account_info_iter)?;
    let user_ai = next_account_info(account_info_iter)?;
    let user_no_ata_ai = next_account_info(account_info_iter)?;
    let no_vault_ai = next_account_info(account_info_iter)?;
    let token_program_ai = next_account_info(account_info_iter)?;
    let system_program_ai = next_account_info(account_info_iter)?;

    if !user_ai.is_signer {
        msg!("User must sign to buy NO");
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

    let market_state = load_market(market_ai)?;
    if market_state.resolved {
        msg!("Market already resolved");
        return Err(ProgramError::InvalidAccountData);
    }
    if *no_vault_ai.key != market_state.no_vault {
        msg!("NO vault mismatch");
        return Err(ProgramError::InvalidAccountData);
    }

    // 1) Transfer SOL from User -> Market
    let transfer_sol_ix =
        system_instruction::transfer(user_ai.key, market_ai.key, amount);
    invoke(
        &transfer_sol_ix,
        &[
            user_ai.clone(),
            market_ai.clone(),
            system_program_ai.clone(),
        ],
    )?;

    // 2) Transfer NO from vault -> user (PDA signs)
    let (pda, bump) =
        Pubkey::find_program_address(&[MARKET_SEED, &market_state.event_id], program_id);
    if pda != *market_ai.key {
        msg!("PDA mismatch while buying NO");
        return Err(ProgramError::InvalidSeeds);
    }

    let transfer_token_ix = token_instruction::transfer(
        token_program_ai.key,
        no_vault_ai.key,
        user_no_ata_ai.key,
        market_ai.key,
        &[],
        amount,
    )?;

    invoke_signed(
        &transfer_token_ix,
        &[
            no_vault_ai.clone(),
            user_no_ata_ai.clone(),
            market_ai.clone(),
            token_program_ai.clone(),
        ],
        &[&[MARKET_SEED, &market_state.event_id, &[bump]]],
    )?;

    msg!("User bought {} NO", amount);
    Ok(())
}

/// Sell NO tokens: user sends NO to vault, receives SOL from Market.
fn process_sell_no(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    amount: u64,
) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();

    let market_ai = next_account_info(account_info_iter)?;
    let user_ai = next_account_info(account_info_iter)?;
    let user_no_ata_ai = next_account_info(account_info_iter)?;
    let no_vault_ai = next_account_info(account_info_iter)?;
    let token_program_ai = next_account_info(account_info_iter)?;
    let system_program_ai = next_account_info(account_info_iter)?;

    if !user_ai.is_signer {
        msg!("User must sign to sell NO");
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

    let market_state = load_market(market_ai)?;
    if market_state.resolved {
        msg!("Market already resolved");
        return Err(ProgramError::InvalidAccountData);
    }
    if *no_vault_ai.key != market_state.no_vault {
        msg!("NO vault mismatch");
        return Err(ProgramError::InvalidAccountData);
    }

    // Ensure Market has enough SOL
    let market_lamports = **market_ai.lamports.borrow();
    if market_lamports < amount {
        msg!("Market has insufficient SOL liquidity");
        return Err(ProgramError::InsufficientFunds);
    }

    // 1) Transfer NO from user -> vault
    let transfer_token_ix = token_instruction::transfer(
        token_program_ai.key,
        user_no_ata_ai.key,
        no_vault_ai.key,
        user_ai.key, // authority = user
        &[],
        amount,
    )?;
    invoke(
        &transfer_token_ix,
        &[
            user_no_ata_ai.clone(),
            no_vault_ai.clone(),
            user_ai.clone(),
            token_program_ai.clone(),
        ],
    )?;

    // 2) Transfer SOL from Market -> User (PDA signs)
    let (pda, bump) =
        Pubkey::find_program_address(&[MARKET_SEED, &market_state.event_id], program_id);
    if pda != *market_ai.key {
        msg!("PDA mismatch while selling NO");
        return Err(ProgramError::InvalidSeeds);
    }

    let transfer_sol_ix =
        system_instruction::transfer(market_ai.key, user_ai.key, amount);
    invoke_signed(
        &transfer_sol_ix,
        &[
            market_ai.clone(),
            user_ai.clone(),
            system_program_ai.clone(),
        ],
        &[&[MARKET_SEED, &market_state.event_id, &[bump]]],
    )?;

    msg!("User sold {} NO", amount);
    Ok(())
}

/// Resolve the market: sets resolved flag and outcome (1 = YES, 2 = NO).
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
