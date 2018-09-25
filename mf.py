"""This file contains customized math/analytics functions"""
import numpy as np
from scipy import stats
#import numexpr as ne


'''
Aggregate all metrics
'''

def compute_metrics(returns):
        results = dict()
        
        returns = np.array(returns)
        cumureturns = cumulative_returns(returns)    
        _drawdown = drawdown(cumureturns)
        
        #Calculates CAGR
        days = len(returns)
        CAGR = cagr(returns, days)
        results['CAGR'] = CAGR
    
        #Calculates volatility 
        Vol = volatility(returns) #assumption: not daily == monthly
        Down_Vol = downside_volatility(returns) #assumption: not daily == monthly
        Up_Vol = upside_volatility(returns) #assumption: not daily == monthly
        results['Volatility'] = (Vol)
        results['Downside Volatility'] = Down_Vol
        results['Upside Volatility'] = Up_Vol
        
        #Calculates max drawdown
        MaxDrawdown = np.nanmin(_drawdown)
        results['Max Drawdown'] = (MaxDrawdown)
    
        #Calculates Sharpe Ratio
        results['Sharpe'] = CAGR/Vol if Vol != 0 else 0
        
        #Calculates Calmar Ratio
        results['Calmar'] = CAGR/-MaxDrawdown if MaxDrawdown != 0 else 0
       
        #Calculates Ulcer Index
        results['Ulcer'] = ulcer_index(CAGR, _drawdown)
    
        #Adding longest recovery
        results['Longest Recovery'] = longest_recovery(_drawdown)
    
        results['Positive Periods'] = positive_periods(returns)
        results['Average Positive'] = average_positive(returns)
        results['Average Negative'] = average_negative(returns)
        
        results['5th Percentile Returns'] = returns_percentile(returns,5)
        results['95th Percentile Returns'] = returns_percentile(returns,95)
        
        #Exponential regression
        results['R Squared'], results["Slope"], _ = exponential_regression(cumureturns)
        
        #Sortino
        results["Sortino"] = sortino(CAGR, Down_Vol)
        
        
        return results


'''
General function
'''

def polyfit(x, y, degree):
    #https://stackoverflow.com/questions/893657/how-do-i-calculate-r-squared-using-python-and-numpy
    results = {}
    coeffs = np.polyfit(x, y, degree)

     # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot

    return results


'''
Financial metrics
'''
def cumulative_returns_last(returns):
    """Returns the (numpy array) cumulative returns of array in percentage terms 
    
    Assumptions: returns are in percentage terms. [0.4, 0.8, 0.1]
    """
    cumprod = np.cumprod(np.array(returns) + 1)[-1]
#    res = ne.evaluate("cumprod - 1")
    return cumprod - 1

def cumulative_returns(returns):
    """Returns the (numpy array) cumulative returns of array in percentage terms 
    
    Assumptions: returns are in percentage terms. [0.4, 0.8, 0.1]
    """
    cumprod_ret = np.cumprod(np.array(returns) + 1) - 1
#    res = 
#    res = ne.evaluate("cumprod - 1")
#    res = ne.evaluate("cumprod - 1")
    return cumprod_ret

def cagr(returns, days):
    """Returns cumulative annual growth rate of returns"""
    cum_returns = cumulative_returns(returns)
    cagr = (cum_returns[-1] + 1)**(365/days) - 1
    
    return cagr


def sharpe(returns, days):
    """Return Sharpe ratio of the array"""
    _cagr = cagr(returns, days)
    vol = volatility(returns)
    sharpe = _cagr/vol if vol != 0 else 0
    
    return sharpe


def volatility(returns, annualized_factor=None):
    """Return daily/monthly volatility of returns, default is daily"""
    days = 252 if annualized_factor is None else annualized_factor
    vol = np.std(returns)*np.sqrt(days)    
    return vol

#unverified
def downside_volatility(returns, annualized_factor=None):
    """Returns the downside volatility of the returns"""    
    days = 252 if annualized_factor is None else annualized_factor
    
    neg_returns = returns[(returns < 0)]
    return volatility(neg_returns, annualized_factor=days)

def upside_volatility(returns, annualized_factor=None):
    """Returns the downside volatility of the returns"""    
    days = 252 if annualized_factor is None else annualized_factor
    
    neg_returns = returns[(returns > 0)]
    return volatility(neg_returns, annualized_factor=days)

def drawdown(cumu_returns):
    """Returns the drawdown of cumulative returns
    
    Assumptions: cumulative is in percentage terms [0.4, 0.8, 0.1]
    """
    cumu_returns = cumu_returns + 1
    cummax = np.maximum.accumulate(cumu_returns)
    
    #TODO: division by zero ?
    res = np.divide(cumu_returns, cummax) - 1
    
    return res
    

def max_drawdown(drawdown):
    """Return the maximum drawdown given a drawdown array"""    
    return np.nanmin(drawdown)
    

def calmar_ratio(cagr, max_drawdown):
    """Returns Calmar Ratio"""
    return cagr/-max_drawdown

#Note: correctness not verified
def ulcer_index(cagr, drawdown):
    """Returns Ulcer Index"""
    denominator = np.sqrt(np.dot(drawdown, drawdown)/len(drawdown))
    return cagr/denominator

def longest_recovery(drawdown):
    """Returns the longest recovery of a drawdown numpy array
    (bit shift approach on longest subsequence problem, on average shows better performance)    
    
    Assumptions: result is the count of longest subsequence of negative number 
    """
    binary_array = np.array(drawdown < 0, dtype=int)
    temp_count = 0
    max_count = 0
    for bit in binary_array:
        if bit == 1:
            temp_count = temp_count + 1
            if temp_count > max_count:
                max_count = temp_count
        else:
            if temp_count > max_count:
                max_count = temp_count
            temp_count = 0
            
    return max_count     

def longest_recovery_2(returns, calculate_percentage = False):
    """Returns the longest recovery of a drawdown numpy array calculated from returns array
    (bit shift approach on longest subsequence problem, on average shows better performance)    
    
    Assumptions: result is the count of longest subsequence of negative number 
    """
    #is there a better approach to get the list of ones ?
    drawdown_array = np.array(drawdown(cumulative_returns(returns)))
    longest_count = longest_recovery(drawdown_array)
    #array = np.concatenate(array).ravel()
    
    if calculate_percentage:
        return longest_count/len(returns)
        
    return longest_count
 
    
def positive_periods(returns):
    """Calculate the percentage of positive days"""
    positive_periods_value = (returns > 0).sum()/(returns != 0).sum()
    return positive_periods_value

def average_positive(returns):
    """Calculate the average return given positive returns"""
    average_positive_value = returns[(returns > 0)].mean()
    return average_positive_value

def average_negative(returns):
    """Calculate the average return given negative returns"""
    average_negative_value = returns[(returns < 0)].mean()
    return average_negative_value

def loss_probability(returns):
    """Calculate the loss probability of returns array"""
    neg_count = (returns < 0).sum()
    return neg_count/len(returns)

#unverified
def skew(returns):
    """Returns the skewness of returns"""
    return stats.skew(returns)

#unverified
def kurtosis(returns):
    """Returns the kurtosis of returns"""
    return stats.kurtosis(returns)

#unverified
def eprpl(mean, loss_prob):
    """Returns mean/loss probability of returns"""
    return mean/loss_prob

#unverified
def sortino(_cagr, down_vol):
    """Returns the Sortino ratio of returns: sortino ratio = cagr/down_vol"""
    return _cagr/down_vol

def returns_percentile(returns, percentile):
    return np.percentile(returns, percentile)


def exponential_regression(cumu_ret, annualized = True):
    L = len(cumu_ret)
    x = np.linspace(1, L, L)
    y = np.log(1+cumu_ret)
    result = polyfit(x,y,1)
    
    rsquared = result["determination"]
    slope = result["polynomial"][0]
    intercept = result["polynomial"][1]
    
    if annualized:
        slope = slope*252
    
    return rsquared, slope, intercept
