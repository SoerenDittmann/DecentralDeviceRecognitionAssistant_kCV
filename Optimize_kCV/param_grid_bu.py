#%% Define Grid
#Extensive params grid
param_grid= {'stc_params': ({
        "estimator": RotationForest(n_estimators=3),
        "n_shapelet_samples": 500, 
        "max_shapelets": 20,
        "batch_size": 100,
    },
    {
        "estimator": RotationForest(n_estimators=2),
        "n_shapelet_samples": 100, 
        "max_shapelets": 5,
        "batch_size": 50,
    }),
    'drcif_params': ({"n_estimators": 10}, 
                     {"n_estimators": 5}),
    'arsenal_params': ({"num_kernels": 50, "n_estimators": 2},
                       {"num_kernels": 100, "n_estimators": 5}),
    'tde_params': ({
        "n_parameter_samples": 25,
        "max_ensemble_size": 5,
        "randomly_selected_params": 10
    },
    {
        "n_parameter_samples": 50,
        "max_ensemble_size": 10,
        "randomly_selected_params": 20
    },
    {
        "n_parameter_samples": 12,
        "max_ensemble_size": 3,
        "randomly_selected_params": 5,
    }),
    'threshold': [0, 0.2, 0.5, 0.8]    
    }