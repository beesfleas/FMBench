from datasets import get_dataset_config_names

try:
    print("GIFT-EVAL configs:", get_dataset_config_names("Salesforce/GiftEval"))
except Exception as e:
    print("Error getting GIFT-EVAL configs:", e)

try:
    print("FEV-Bench configs:", get_dataset_config_names("autogluon/fev_datasets"))
except Exception as e:
    print("Error getting FEV-Bench configs:", e)
