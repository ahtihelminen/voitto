import pandas as pd

# Constants for Standard Vig (-110 American Odds)
STANDARD_DECIMAL_ODDS = 1.909  # 1 + 100/110
BREAK_EVEN_PROB = 0.5238       # 110 / 210

def calculate_kelly_stake(
    prob_win: float, 
    decimal_odds: float, 
    kelly_fraction: float
) -> float:
    """
    Calculates the Kelly Criterion stake percentage.
    f* = (bp - q) / b
    where:
        b = net odds received on wager (decimal - 1)
        p = probability of winning
        q = probability of losing (1 - p)
    """
    if prob_win <= 0:
        return 0.0
        
    b = decimal_odds - 1
    p = prob_win
    q = 1 - p
    
    f_star = (b * p - q) / b
    
    # We only bet if edge is positive (f_star > 0)
    return max(0.0, f_star) * kelly_fraction

def size_bets(
    predictions: pd.DataFrame,
    bankroll: float,
    kelly_fraction: float = 0.25,
    min_edge: float = 0.0,
) -> pd.DataFrame:
    """
    Takes predictions with 'prob_over' and assigns bets.
    
    Returns DataFrame with added columns:
        - bet_side: 'Over', 'Under', or 'Pass'
        - bet_size: Dollar amount to wager
        - edge: The theoretical edge %
    """
    predictions_df = predictions.copy()
    
    bet_sides = []
    bet_sizes = []
    edges = []
    
    for _, row in predictions_df.iterrows():
        prob_over = row["prob_over"]
        prob_under = 1 - prob_over
        
        # Determine preferred side
        if prob_over > BREAK_EVEN_PROB + min_edge:
            side = "Over"
            prob_win = prob_over
        elif prob_under > BREAK_EVEN_PROB + min_edge:
            side = "Under"
            prob_win = prob_under
        else:
            side = "Pass"
            prob_win = 0.0
            
        # Calculate Stake
        if side != "Pass":
            stake_pct = calculate_kelly_stake(
                prob_win, STANDARD_DECIMAL_ODDS, kelly_fraction
            )
            wager = bankroll * stake_pct
            edge = prob_win - BREAK_EVEN_PROB
        else:
            wager = 0.0
            edge = 0.0
            
        bet_sides.append(side)
        bet_sizes.append(wager)
        edges.append(edge)
        
    predictions_df["bet_side"] = bet_sides
    predictions_df["bet_amount"] = bet_sizes
    predictions_df["edge_pct"] = edges
    
    return predictions_df