def create_config_template(file_path):
    """
    Creates a configuration template file from a given JSON data.

    Args:
    json_data (dict): A dictionary representing the JSON configuration for a TimeSeriesDataSet.
    file_path (str): The path where the configuration template file will be saved.

    This function reads a JSON dictionary, representing the configuration for a TimeSeriesDataSet,
    and writes it to a file as a template. The file can then be edited by users to fit their specific needs.
    """
    import json

    # Example JSON data
    json_example = {
        "time_idx": "time_idx",
        "target": ["target"],
        "group_ids": ["county", "is_business", "product_type", "is_consumption"],
        "min_encoder_length": 12,
        "max_encoder_length": 24,
        "min_prediction_length": 1,
        "max_prediction_length": 1,
        "static_categoricals": ["county", "is_business", "product_type", "is_consumption"],
        "static_reals": ["eic_count", "installed_capacity"],
        "lags": {},
        "time_varying_known_categoricals": ["month", "day", "dayofweek", "dayofyear", "hour"],
        "variable_groups": {},
        "time_varying_known_reals": [
            "time_idx", "lowest_price_per_mwh", "highest_price_per_mwh", "euros_per_mwh_forecat",
            "temperature_forecast", "dewpoint_forecast", "cloudcover_total_forecast",
            "10_metre_u_wind_component_forecast", "10_metre_v_wind_component_forecast",
            "direct_solar_radiation_forecast", "surface_solar_radiation_downwards_forecast",
            "snowfall_forecast", "total_precipitation_forecast"
        ],
        "time_varying_unknown_categoricals": [],
        "time_varying_unknown_reals": ["target", "log_target"],
        "scalers": {"StandardScaler": ["Import", "Export"]},
        "add_relative_time_idx": True,
        "add_target_scales": True,
        "add_encoder_length": True,
        "allow_missing_timesteps": True,
        "categorical_encoders": ["county", "is_business", "product_type", "is_consumption"],
        "target_normalizer": ["county", "is_business", "product_type", "is_consumption"]
    }

    # Writing the JSON data to a file
    with open(file_path, 'w') as file:
        json.dump(json_example, file, indent=4)


def get_categorical_variables(config: dict[str: any]):
    return set(config['group_ids'] + config['static_categoricals'] + config['time_varying_known_categoricals'] + config['time_varying_unknown_categoricals'])

def get_numerical_variables(config: dict[str: any]):
    return [col for col in set(config["time_varying_known_reals"] + config["static_reals"] + config["time_varying_unknown_reals"]) if col not in config['target']]
