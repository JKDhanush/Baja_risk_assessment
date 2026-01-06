MODEL_REGISTRY = {
    "risk_classifier": {
        "inputs": ["price_volatility", "lead_time", "supplier_dependency", "impact"],
        "output": "Overall Risk Level"
    },
    "delay_predictor": {
        "inputs": ["lead_time", "logistics_uncertainty", "buffer_days"],
        "output": "Delay Probability (%)"
    },
    "cost_overrun_predictor": {
        "inputs": ["price_volatility", "emergency_procurement", "buffer_budget"],
        "output": "Cost Overrun Probability (%)"
    }
}
