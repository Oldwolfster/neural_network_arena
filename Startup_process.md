# Startup process 4.30.2025

| Step | Stage               | Component        | Method                        | Description                                                                                  | Config State               |
|------|---------------------|------------------|-------------------------------|----------------------------------------------------------------------------------------------|----------------------------|
| 1    | Prep                | Engine           | engine Constructor            | Creates training data                                                                        |                            |
| 2    | Prep                | Engine           | run_a_match                   | Gets next gladiator name                                                                     |                            |
| 3    | Prep                | Engine           | run_a_match                   | Sets or resets random seed                                                                   |                            |
| 4    | Prep                | Engine           | run_a_match                   | Calls `atomic_train_a_model` to train that gladiator                                         |                            |
| 5    | Prep                | engine           | Atomic_train_a_model          | Creates weight adjustment tables                                                             |                            |
| 6    | Prep                | engine           | Atomic_train_a_model          | Calls `check_for_learning_rate_sweep` (or uses specified)                                    |                            |
| 7    | Prep                | engine           | check_for_learning_rate_sweep| Creates `check_config` to test config (🔄 reused later!)                                     | check_config created       |
| 8    | Prep                | engine           | check_for_learning_rate_sweep| Calls `set_defaults(check_config)`                                                           | input_scaler assigned?     |
| 9    | Prep                | Config           | set_defaults                  | Calls `smartNetworkSetup` — 🤔 but subclass info not available yet                           |                            |
| 10   | Prep                | Config           | set_defaults                  | Passes training data to input_scaler                                                        | input_scaler.fit(data)     |
| 11   | Instantiate gladiator | engine         | check_for_learning_rate_sweep| Instantiates gladiator w/ `check_config`                                                     |                            |
| 12   | Instantiate gladiator | base_gladiator | constructor                   | Calls `retrieve_setup_from_model`                                                            |                            |
| 13   | Instantiate gladiator | base_gladiator | retrieve_setup_from_model     | Delegates to subclass to call `configure_model`                                              |                            |
| 14   | Instantiate gladiator | sub_gladiator  | configure_model               | Can set anything or nothing in config (pre-neuron settings)                                  | May override input_scaler? |
| 15   | Instantiate gladiator | base_gladiator | retrieve_setup_from_model     | Calls `config.configure_optimizer()`                                                         |                            |
| 16   | Instantiate gladiator | config         | configure_optimizer           | Sets optimizer — affects popup headers                                                      |                            |
| 17   | Instantiate gladiator | base_gladiator | retrieve_setup_from_model     | Calls `initialize_neurons`                                                                  |                            |
| 18   | Instantiate gladiator | base_gladiator | retrieve_setup_from_model     | Delegates to subclass for post-neuron creation setup                                         |                            |
| 19   | LR Sweep            | engine           | check_for_learning_rate_sweep| If LR is defined in subclass, returns it early                                               |                            |
| 20   | LR Sweep            | engine           | check_for_learning_rate_sweep| Calls actual sweep method                                                                    |                            |
| 21   | LR Sweep            | engine           | learning_rate_sweep           | Loops through learning rates                                                                 | check_config reused        |
| 22   | LR Sweep            | engine           | learning_rate_sweep           | 🔁 Re-instantiates same gladiator using `check_config` (steps 12–18)                         | input_scaler reused?       |
| 23   | LR Sweep            | engine           | learning_rate_sweep           | Returns best LR to `atomic_train_a_model`                                                    | best_lr chosen             |
| 24   | Train for real      | engine           | Atomic_train_a_model          | Marks time, creates fresh `model_config` ⚠️                                                 | new model_config created   |
| 25   | Train for real      | engine           | Atomic_train_a_model          | Resets seed                                                                                  |                            |
| 26   | Train for real      | engine           | Atomic_train_a_model          | Re-instantiates gladiator w/ new `model_config` (steps 12–18 again)                         |                            |
| 27   | Train for real      | model_Config     | set_defaults                  | Calls `smartNetworkSetup` again — 🤔 subclass config timing?                                 |                            |
| 28   | Train for real      | model_Config     | set_defaults                  | Passes training data to input_scaler                                                        | input_scaler.fit() again?  |
| 29   | Train for real      | engine           | Atomic_train_a_model          | Finally: train the model! Initialization complete                                            |                            |
