# sacco_core/analytics/pricing.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime

class PricingOptimizer:
    """Optimize loan and deposit pricing for maximum profitability"""
    
    def __init__(self):
        self.product_parameters = self._initialize_product_parameters()
        self.market_data = self._initialize_market_data()
    
    def _initialize_product_parameters(self) -> Dict[str, Dict]:
        """Initialize default product parameters"""
        return {
            "Personal Loan": {
                "base_operating_cost": 0.018,
                "risk_weight": 1.0,
                "elasticity": -1.2,
                "min_volume": 1000000
            },
            "Business Loan": {
                "base_operating_cost": 0.025,
                "risk_weight": 1.3,
                "elasticity": -0.8,
                "min_volume": 5000000
            },
            "Asset Finance": {
                "base_operating_cost": 0.015,
                "risk_weight": 0.9,
                "elasticity": -1.0,
                "min_volume": 2000000
            },
            "Emergency Loan": {
                "base_operating_cost": 0.030,
                "risk_weight": 1.8,
                "elasticity": -0.5,
                "min_volume": 500000
            },
            "School Fees": {
                "base_operating_cost": 0.012,
                "risk_weight": 0.7,
                "elasticity": -1.5,
                "min_volume": 500000
            }
        }
    
    def _initialize_market_data(self) -> Dict[str, float]:
        """Initialize market benchmark data"""
        return {
            "Personal Loan": 0.145,
            "Business Loan": 0.160,
            "Asset Finance": 0.135,
            "Emergency Loan": 0.190,
            "School Fees": 0.110,
            "Savings Account": 0.038,
            "Fixed Deposit 30D": 0.055,
            "Fixed Deposit 90D": 0.070,
            "Fixed Deposit 180D": 0.078,
            "Fixed Deposit 1Y": 0.085
        }
    
    def calculate_optimal_pricing(self,
                                product_type: str,
                                loan_amount: float,
                                loan_term: int,
                                risk_category: str,
                                cost_of_funds: float,
                                operating_cost: float,
                                target_roa: float,
                                risk_premium: float) -> Dict[str, Any]:
        """
        Calculate optimal loan pricing
        
        Args:
            product_type: Type of loan product
            loan_amount: Loan amount in KES
            loan_term: Loan term in months
            risk_category: Borrower risk category
            cost_of_funds: Cost of funds as decimal
            operating_cost: Operating cost as decimal
            target_roa: Target return on assets as decimal
            risk_premium: Additional risk premium as decimal
            
        Returns:
            Dictionary with optimal pricing results
        """
        # Get product-specific parameters
        product_params = self.product_parameters.get(product_type, {})
        
        # Calculate base break-even rate
        break_even_rate = self._calculate_break_even_rate(
            cost_of_funds, operating_cost, product_params.get('base_operating_cost', 0)
        )
        
        # Calculate risk provision based on category
        risk_provision = self._calculate_risk_provision(risk_category, loan_term)
        
        # Calculate target return component
        return_component = target_roa * (12 / loan_term)  # Annualize for loan term
        
        # Calculate recommended rate
        recommended_rate = (
            break_even_rate +
            risk_provision +
            return_component +
            risk_premium
        )
        
        # Apply product-specific adjustments
        recommended_rate *= product_params.get('risk_weight', 1.0)
        
        # Ensure rate is within reasonable bounds
        recommended_rate = max(recommended_rate, break_even_rate * 1.1)
        recommended_rate = min(recommended_rate, break_even_rate * 2.5)
        
        # Check market competitiveness
        market_rate = self.market_data.get(product_type, recommended_rate)
        market_competitive = recommended_rate <= market_rate * 1.1
        
        # Calculate expected profitability
        risk_adjusted_return = self._calculate_risk_adjusted_return(
            recommended_rate, break_even_rate, risk_category
        )
        
        return {
            'product_type': product_type,
            'loan_amount': loan_amount,
            'loan_term': loan_term,
            'risk_category': risk_category,
            'recommended_rate': recommended_rate,
            'break_even_rate': break_even_rate,
            'risk_provision': risk_provision,
            'target_roa': target_roa,
            'risk_premium': risk_premium,
            'cost_of_funds': cost_of_funds,
            'operating_cost': operating_cost,
            'market_rate': market_rate,
            'market_competitive': market_competitive,
            'risk_adjusted_return': risk_adjusted_return,
            'expected_profit': loan_amount * (recommended_rate - break_even_rate) * (loan_term / 12),
            'calculation_date': datetime.now().strftime('%Y-%m-%d')
        }
    
    def _calculate_break_even_rate(self, 
                                 cost_of_funds: float, 
                                 operating_cost: float,
                                 product_operating_cost: float) -> float:
        """Calculate break-even interest rate"""
        total_operating_cost = operating_cost + product_operating_cost
        return cost_of_funds + total_operating_cost
    
    def _calculate_risk_provision(self, risk_category: str, loan_term: int) -> float:
        """Calculate risk provision based on category and term"""
        base_risk_rates = {
            "Low": 0.005,
            "Medium": 0.015,
            "High": 0.035,
            "Very High": 0.060
        }
        
        base_rate = base_risk_rates.get(risk_category, 0.015)
        
        # Adjust for loan term (longer terms = higher risk)
        term_adjustment = max((loan_term - 12) / 12 * 0.005, 0)
        
        return base_rate + term_adjustment
    
    def _calculate_risk_adjusted_return(self, 
                                      recommended_rate: float, 
                                      break_even_rate: float,
                                      risk_category: str) -> float:
        """Calculate risk-adjusted return"""
        gross_return = recommended_rate - break_even_rate
        
        # Risk adjustment factors
        risk_factors = {
            "Low": 1.0,
            "Medium": 0.85,
            "High": 0.65,
            "Very High": 0.45
        }
        
        risk_factor = risk_factors.get(risk_category, 0.75)
        
        return gross_return * risk_factor
    
    def sensitivity_analysis(self, base_result: Dict[str, Any]) -> Dict[str, Dict]:
        """
        Perform sensitivity analysis on pricing components
        
        Args:
            base_result: Base pricing calculation result
            
        Returns:
            Dictionary with sensitivity analysis results
        """
        sensitivities = {}
        
        # Cost of funds sensitivity
        cost_variations = np.linspace(-0.02, 0.02, 5)  # ±2%
        cost_rates = []
        
        for variation in cost_variations:
            new_cost = base_result['cost_of_funds'] + variation
            new_rate = self.calculate_optimal_pricing(
                product_type=base_result['product_type'],
                loan_amount=base_result['loan_amount'],
                loan_term=base_result['loan_term'],
                risk_category=base_result['risk_category'],
                cost_of_funds=new_cost,
                operating_cost=base_result['operating_cost'],
                target_roa=base_result['target_roa'],
                risk_premium=base_result['risk_premium']
            )['recommended_rate']
            cost_rates.append(new_rate)
        
        sensitivities['cost_of_funds'] = {
            'values': cost_variations,
            'rates': cost_rates
        }
        
        # Operating cost sensitivity
        opcost_variations = np.linspace(-0.01, 0.01, 5)  # ±1%
        opcost_rates = []
        
        for variation in opcost_variations:
            new_opcost = base_result['operating_cost'] + variation
            new_rate = self.calculate_optimal_pricing(
                product_type=base_result['product_type'],
                loan_amount=base_result['loan_amount'],
                loan_term=base_result['loan_term'],
                risk_category=base_result['risk_category'],
                cost_of_funds=base_result['cost_of_funds'],
                operating_cost=new_opcost,
                target_roa=base_result['target_roa'],
                risk_premium=base_result['risk_premium']
            )['recommended_rate']
            opcost_rates.append(new_rate)
        
        sensitivities['operating_cost'] = {
            'values': opcost_variations,
            'rates': opcost_rates
        }
        
        # Risk premium sensitivity
        risk_variations = np.linspace(-0.03, 0.03, 5)  # ±3%
        risk_rates = []
        
        for variation in risk_variations:
            new_risk = base_result['risk_premium'] + variation
            new_rate = self.calculate_optimal_pricing(
                product_type=base_result['product_type'],
                loan_amount=base_result['loan_amount'],
                loan_term=base_result['loan_term'],
                risk_category=base_result['risk_category'],
                cost_of_funds=base_result['cost_of_funds'],
                operating_cost=base_result['operating_cost'],
                target_roa=base_result['target_roa'],
                risk_premium=new_risk
            )['recommended_rate']
            risk_rates.append(new_rate)
        
        sensitivities['risk_premium'] = {
            'values': risk_variations,
            'rates': risk_rates
        }
        
        return sensitivities
    
    def optimize_deposit_rates(self, 
                              target_deposit_growth: float,
                              current_deposit_mix: Dict[str, float],
                              market_rates: Dict[str, float]) -> Dict[str, float]:
        """
        Optimize deposit rates to achieve growth targets
        
        Args:
            target_deposit_growth: Target deposit growth rate
            current_deposit_mix: Current deposit product mix
            market_rates: Current market rates for deposit products
            
        Returns:
            Dictionary with optimized deposit rates
        """
        optimized_rates = {}
        
        for product, current_rate in current_deposit_mix.items():
            market_rate = market_rates.get(product, current_rate)
            
            # Strategic positioning
            if product == 'Savings Account':
                # Be competitive on savings to attract core deposits
                optimized_rates[product] = min(market_rate * 1.05, current_rate * 1.02)
            elif 'Fixed Deposit' in product:
                # Optimize fixed deposits based on term
                if '30D' in product:
                    optimized_rates[product] = market_rate * 0.98  # Slightly below market for short-term
                elif '1Y' in product:
                    optimized_rates[product] = market_rate * 1.02  # Above market for long-term
                else:
                    optimized_rates[product] = market_rate  # Match market for medium-term
            else:
                optimized_rates[product] = market_rate
        
        return optimized_rates
    
    def calculate_portfolio_impact(self,
                                 current_rates: Dict[str, float],
                                 proposed_rates: Dict[str, float],
                                 product_volumes: Dict[str, float],
                                 price_elasticities: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate impact of rate changes on portfolio
        
        Args:
            current_rates: Current interest rates
            proposed_rates: Proposed new rates
            product_volumes: Current product volumes
            price_elasticities: Price elasticity for each product
            
        Returns:
            Dictionary with portfolio impact analysis
        """
        total_impact = 0
        volume_changes = {}
        revenue_changes = {}
        
        for product in current_rates.keys():
            current_rate = current_rates[product]
            proposed_rate = proposed_rates[product]
            current_volume = product_volumes.get(product, 0)
            elasticity = price_elasticities.get(product, -1.0)
            
            # Calculate volume change
            rate_change_pct = (proposed_rate - current_rate) / current_rate
            volume_change_pct = rate_change_pct * elasticity
            new_volume = current_volume * (1 + volume_change_pct)
            
            # Calculate revenue change
            current_revenue = current_volume * current_rate
            new_revenue = new_volume * proposed_rate
            revenue_change = new_revenue - current_revenue
            
            volume_changes[product] = {
                'current_volume': current_volume,
                'new_volume': new_volume,
                'volume_change_pct': volume_change_pct * 100
            }
            
            revenue_changes[product] = {
                'current_revenue': current_revenue,
                'new_revenue': new_revenue,
                'revenue_change': revenue_change
            }
            
            total_impact += revenue_change
        
        return {
            'total_impact': total_impact,
            'volume_changes': volume_changes,
            'revenue_changes': revenue_changes,
            'overall_impact_pct': (total_impact / sum(product_volumes.values())) * 100
        }