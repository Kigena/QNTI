
# Add this to your qnti_main_system.py in the _setup_web_interface method:

# Initialize Forex Financial Advisor
try:
    from qnti_forex_financial_advisor import integrate_forex_advisor_with_qnti
    self.forex_advisor = integrate_forex_advisor_with_qnti(self)
    if self.forex_advisor:
        logger.info("Forex Financial Advisor integrated successfully")
    else:
        logger.warning("Forex Financial Advisor integration failed")
except ImportError:
    logger.info("Forex Financial Advisor not available (module not found)")
except Exception as e:
    logger.warning(f"Forex Financial Advisor integration failed: {e}")
