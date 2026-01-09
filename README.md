# FinGrowth

## Data Generation

The `data_generation.py` file builds a synthetic personal-finance dataset for a 1-person household. It produces 6 income clusters (`C1_low`, `C2_lower_mid`, `C3_mid`, `C4_upper_mid`, `C5_high`, and `C6_top5`), with each person having 48 months of data and each month represented by a single row with 17 columns (Income with 16 outflow categories). The datasets are generated as 6 CSV files (one per income cluster), along with a `generation_metadata.json` file that captures the run settings and the exact column order. The generator is designed to be constraint-aware, with realistic bounds; diverse, so that nothing collapses to the average; and reproducible, with seeded runs and metadata.

### Initial data source:
* Data report source: [United States Census Bureau](https://www2.census.gov/library/publications/2025/demo/p60-286.pdf)
* Generated data set download link: [DataSets](https://mailuc-my.sharepoint.com/:f:/g/personal/tikhonam_mail_uc_edu/IgBEzX8vOTpVRZjoJVSQTCKmAWKpCWVlypzDgvrUqe8SFDA?e=JKGOxa)

### Clusters
1. Cluster 1 - Financially strained (bottom 20%): ≤ $34,510 (around < $2,300/mo)
2. Cluster 2 - Working / lower-middle (20–40%): $34,511 – $65,100 (around $2,300 – $4,200/mo)
3. Cluster 3 - Middle / stable (40–60%): $65,101 – $105,500 (around $4,200 – $6,200/mo)
4. Cluster 4 - Upper-middle (60–80%): $105,501 – $175,700 (around $6,200 – $12,900/mo)
5. Cluster 5 - Affluent (80–95%): $175,701 – $335,699 (around $12,900 – $17,500/mo)
6. Cluster 6 - Top 5%: ≥ $335,700 (around ≥ $17,500/mo)

### Data columns
* Income / deposits (`Income_Deposits`): paycheck, benefits/unemployment, tax refund, interest/dividends, P2P received, cash deposit, and employer reimbursements.
* Housing (`Housing`): rent, mortgage payment, HOA/condo fees, property tax, renter/home supplies, home maintenance, furniture, and moving/storage.
* Utilities (`Utilities_Telecom`): electric, gas utility, water/sewer, trash, internet, mobile phone, streaming TV/internet add-ons, and home security monitoring.
* Groceries (`Groceries_FoodAtHome`): supermarket, wholesale club, convenience store snacks, meal kits, specialty food shops, bakery, coffee beans/at-home, and alcohol at store.
* Dining / delivery (`Dining_FoodAway`): restaurants, fast food, cafes, bars, food delivery apps, tips, work lunch, and catering.
* Daily transportation (`Transportation_Variable`): gas/fuel, public transit, rideshare, parking, tolls, bike/scooter rental, and commuter rail/bus pass.
* Auto costs ownership (`Auto_Costs`): car payment/lease, insurance premium, repairs, maintenance/oil, tires, registration/DMV, car wash, and roadside assistance.
* Healthcare (`Healthcare_OOP`): doctor visits, dental, vision, prescriptions, therapy, medical devices/supplies, lab tests, and copays/deductibles.
* Insurance (`Insurance_All`): renters insurance, homeowners insurance, life insurance, disability insurance, umbrella policy, health premium, and pet insurance.
* Debt payments (`Debt_Payments`): credit card payment, student loan payment, personal loan, BNPL installments, payday/advance repayment, interest/fees, and collections/charge-off payments.
* Savings / investments (`Savings_Investments`): brokerage deposits, 401k/403b contribution, IRA/Roth contribution, HSA contribution, savings transfer, CDs/treasuries purchase, and crypto purchase.
* Education / childcare (`Education_Childcare`): tuition, student fees, books/supplies, online courses, daycare, babysitting, after-school programs, and tutoring.
* Entertainment / hobbies (`Entertainment`): movies, concerts/events, gaming, hobbies/crafts, sports/fitness classes, museums/parks, lottery, and nightlife.
* Subscriptions / memberships (`Subscriptions_Memberships`): streaming subscriptions, music subscriptions, app/software subscriptions, gym membership, warehouse membership, newsletters, cloud storage, and domain/web hosting.
* Cash / ATM / misc transfers (`Cash_ATM_MiscTransfers`): ATM withdrawals, cash back, P2P sent, wire transfers, bank fees, money orders, currency exchange, and uncategorized merchant transactions.
* Pets (`Pets`): pet food, vet visits, grooming, boarding/daycare, toys/treats, training, pet medications, pet supplies, and adoption fees.
* Travel (`Travel`): airfare, hotel/lodging, car rental, trains/buses, rideshare while traveling, travel insurance, baggage/fees, tours/activities, visas/passports, and souvenirs.

### What the script generates
With default settings `--people-per-cluster 20000 --months 48 --include-metadata`:
1. Clusters: 6 (`C1_low`, `C2_lower_mid`, `C3_mid`, `C4_upper_mid`, `C5_high`, and `C6_top5`)
2. Rows per cluster file: `20,000` people × `48` months = `960,000` rows
3. Total rows across all clusters: `6` × `960,000` = `5,760,000` rows
4. Files produced (download link):
    * `DataSets/C1_low.csv`
    * `DataSets/C2_lower_mid.csv`
    * `DataSets/C3_mid.csv`
    * `DataSets/C4_upper_mid.csv`
    * `DataSets/C5_high.csv`
    * `DataSets/C6_top5.csv`
    * `DataSets/generation_metadata.json` (the run metadata: `seeds`, `counts`, `columns`, etc.)
5. Each CSV row represents one person in one month, and (optionally) includes identifiers: `cluster`, `person_id`, `month_index` (enabled via `--include-metadata`)

### Why these income clusters and percentage ranges exist
1. Income cluster cutoffs are modeled after U.S. income distribution tables (quintiles and upper tail), then treated as individual income for a 1-person household for this project’s simplified assumption.
    * Source: [Census.gov](https://www.census.gov/library/publications/2023/demo/p60-279.html)
2. Spending-share ranges (the percentage of income clamps) are inspired by how spending categories vary across income groups in national consumer expenditure statistics. The generator intentionally keeps these ranges wide so the dataset covers many plausible behaviors, while the Dirichlet, traits, and constraints functions create the realistic shape.
    * Source: [Bureau of Labor Statistics - U.S. Department of Labor](https://www.bls.gov/news.release/archives/cesan_09252024.pdf)
3. The generator also encodes practical budgeting, like housing stress thresholds (for example, cost burden commonly referenced at ~30% of income), which is why lower-income housing percentage bands are higher.
    * Source: [HUD.GOV U.S. Department of Housing and Urban Development](https://www.huduser.gov/archives/portal/pdredge/pdr-edge-featd-article-091724.html)

### How the generator works
1. **Define 6 income clusters (annual ranges)**

    The script starts with 6 income bands (1CLUSTERS1) and treats them as annual income ranges for a 1-person household. Cluster 6 (top 5%) uses a practical max bound (default `1,500,000`) so sampling does not explode, and it is adjustable via `--top5-max`.

2. **Fix the schema: 17 columns (1 income with 16 outflows)**

    This part matches both the UI's needs and the ML pipeline's expectations for the income column (`Income_Deposits`) and 16 spending/outflow categories (housing, groceries, debt payments, savings/investments, travel, etc.).

3. **Set hard monthly floors/caps per category (`ABS_BOUNDS`)**

    These are absolute guardrails that prevent unrealistic values like negative groceries, $0 housing for everyone, or $200k/month subscriptions. They keep the data numerically stable and realistic, even when income is very low or very high.

4. **Set soft cluster-specific percentage ranges (`PCT_RANGES`)**

    For each cluster and each category, the script defines a percentage of income range (low/high). These are applied as dynamic clamps that scale with monthly income:
    * `floor = max(abs_floor, pct_lo * income_monthly)`
    * `cap = max(abs_cap, pct_hi * income_monthly)`

    This is the main mechanism that makes each cluster different. For example, housing share tends to be higher among low-income households, and saving capacity increases with income.

5. **Sample person-level traits with Latin Hypercube Sampling (LHS)**

    Instead of naive random sampling, which tends to clump around the mean, LHS spreads people evenly across the trait space. That means reliably get a mix of:
    * savers vs spenders
    * debt-heavy vs debt-light
    * travel-heavy, dining-heavy, cash-heavy patterns
    * car owners / pet owners

    This improves coverage and makes the Machine Learning pipeline training set more robust.

6. **Build a person template using a Dirichlet prior (category `share` model)**

    Each person gets a Dirichlet alpha vector across 16 categories:
    * baseline alpha comes from the midpoints of `PCT_RANGES`
    * then traits nudge certain categories up/down (e.g., travel-heavy boosts Travel alpha)

    Dirichlet was chosen because it is a clean, standard way to generate proportions that always sum to 1 (perfect fit for the `spending composition`).

7. **Simulate 48 months per person with realistic month-to-month variation**

    For each month:
    * income varies slightly (noise), with rare bonus months (more likely for high clusters)
    * total outflows are set by a sampled outflow ratio (balanced vs debt-stress vs aggressive-saver modes)
    * category shares are drawn from the Dirichlet (composition varies naturally)
    * seasonality and spikes are applied (utilities seasonality, travel spikes, healthcare events, auto repairs)

8. **Enforce constraints and close the budget via rebalancing**

    After generating raw amounts:
    * the script clamps each category to its dynamic and income-scaled floor/cap
    * then it rebalances amounts so the sum matches the target total outflow as closely as possible while respecting constraints

    This is why the data is reliable for training/testing: every row is internally consistent and falls within plausible bounds.

9. **Write large CSV files with chunked streaming**

    The generator writes rows incrementally in `chunk_people` blocks, so it can scale to millions of rows without excessive RAM usage. The output is deterministic within each cluster due to `cluster_seed`.

## Data Pipeline & Model Training

This project trains a supervised model to predict a person's financial group (`C1_low`, `C2_lower_mid`, `C3_mid`, `C4_upper_mid`, `C5_high`, and `C6_top5`) from monthly personal-finance records. The raw data is month-level (48 months per person), but the model is trained on person-level engineered features such as shares/rates and aggregation across months to avoid month-to-month noise and reduce leakage. Below is the pipeline structure from the notebook (`data_model_pipeline.ipynb`), with what it does and why it is there.

1. **Setup**

    The notebook starts by importing the usual stack (NumPy/Pandas/Sklearn/XGBoost/Matplotlib) and setting a consistent random seed. This is mostly about reproducibility and making sure that comparisons between preprocessing methods are fair (same split, same randomization, and same evaluation metric).

2. **Load the 6 cluster datasets with month-level to person-level**

    Load the six synthetic CSV files (`C1_low`, `C2_lower_mid`, `C3_mid`, `C4_upper_mid`, `C5_high`, and `C6_top5`) and map them to labels `0`, `1`, `2`, `3`, `4`, and `5`. Each file corresponds to a single cluster, so the task becomes a multi-class classification problem.

    **Key point:** each person has 48 rows (one row per month). Splitting must happen at the person level, not the row level, otherwise months from the same person can leak into both the train and test sets.

3. **Helper functions (plots, splits, quick eval, and STATE)**

    The notebook defines utility functions that make the pipeline modular:

    * Person-level splits using `GroupShuffleSplit` (group = `person_id`). This prevents leakage and makes validation more realistic, since the model must generalize to new people rather than new months from the same person.
    * Quick evaluation helpers that compute accuracy, macro-F1, and generate confusion matrices / summary tables.
    * A lightweight STATE dict used to carry the current dataset/features through each pipeline step while comparing alternatives.

4. **Data Cleaning**

    This stage makes the month-level table safe to engineer features from:
    * Replace non-finite values ($±inf$) with $NaN$, then drop rows that are unusable, especially missing or non-positive income. Most features are a share of income, so income must be valid.
    * Fill missing outflow amounts as $0$ (practical assumption for synthetic data).
    * Create totals that later become rate features or examples:
        * `TotalOutflows = sum(outflows)`
        * `NetCashflow = Income - TotalOutflows`

        This makes it easy to compute net-flow style ratios later.

5. **Feature Encoding**

    Month-level engineered features (baseline eval (LR): acc = `0.9997` and macro_f1= `0.9997`)

    * For each month $t$ and category $c$:
        * Share of income: $share_{c,t} = \frac{outflow_{c,t}}{income_t}$

        * Rates:
            * `TotalOutflowRate = TotalOutflows / Income`
            * `SavingsRate = Savings_Investments / Income`
            * `DebtRate = Debt_Payments / Income`
            * `EssentialsRate = (Housing + Utilities + Groceries) / Income`
            * `DiscretionaryRate = (Dining + Entertainment + Subscriptions + Travel) / Income`
            * `NetCashflowRate = NetCashflow / Income`

    * Person-level aggregation (48 months into 1 row per person)

        Group by `person_id` and aggregate each engineered feature with: $mean$, $median$, and $std$. This provides a stable fingerprint of behavior over time, rather than letting one unusual month dominate.

    * Benefits of this design:
        * Shares/rates reduce sensitivity to absolute income scale.
        * Aggregation reduces noise.
        * $Mean/median/std$ capture both typical behavior and volatility.

6. **Cluster checks**

    Before running model comparisons, the notebook includes basic checks to avoid training on bad data:
    * Do the labels look balanced or at least expected?
    * Do engineered rates stay in plausible ranges?
    * Do per-cluster distributions differ enough to be learnable?

7. **Preprocessing steps**

    Each step below is tested by training a simple baseline model and selecting the variant with the best macro-F1, as macro-F1 treats all clusters equally and avoids being dominated by the easiest class.

    7.1 **Normalization (best = `sqrt`)**

    * **`none`**

        No transform: $x' = x$. Keeps original units; good baseline, but skewed heavy-tail spending can dominate learning.
        <p>
            <img src="./Data%20Visualizations/Normalization%20(none_1).png" height="100%" width="33%" />
            <img src="./Data%20Visualizations/Normalization%20(none_2).png" height="100%" width="30%" />
            <img src="./Data%20Visualizations/Normalization%20(none_3).png" height="100%" width="30%" />
        </p>


    * **`log`**

        $x' = log(1 + x)$. Compresses large values and reduces right-skew so high spenders do not overwhelm the model.
        <p>
            <img src="./Data%20Visualizations/Normalization%20(log_1).png" height="100%" width="33%" />
            <img src="./Data%20Visualizations/Normalization%20(log_2).png" height="100%" width="30%" />
            <img src="./Data%20Visualizations/Normalization%20(log_3).png" height="100%" width="30%" />
        </p>

    * **`inverse`**

        $x' = \frac{1}{x + \epsilon}$. Down-weights large magnitudes and highlights small-but-nonzero values; very sensitive near zero, so $\epsilon$ is required.
        <p>
            <img src="./Data%20Visualizations/Normalization%20(inverse_1).png" height="100%" width="33%" />
            <img src="./Data%20Visualizations/Normalization%20(inverse_2).png" height="100%" width="30%" />
            <img src="./Data%20Visualizations/Normalization%20(inverse_3).png" height="100%" width="30%" />
        </p>

    * **`square`**

        $x' = x^2$. Expands large values and increases separation for high spend behavior; can worsen skew and outliers.
        <p>
            <img src="./Data%20Visualizations/Normalization%20(square_1).png" height="100%" width="33%" />
            <img src="./Data%20Visualizations/Normalization%20(square_2).png" height="100%" width="30%" />
            <img src="./Data%20Visualizations/Normalization%20(square_3).png" height="100%" width="30%" />
        </p>

    * **`zscore`**

        $x' = \frac{x - \mu}{\sigma}$. Centers and scales each feature to comparable units; assumes mean/std are meaningful and can be outlier-sensitive.
        <p>
            <img src="./Data%20Visualizations/Normalization%20(zscore_1).png" height="100%" width="33%" />
            <img src="./Data%20Visualizations/Normalization%20(zscore_2).png" height="100%" width="30%" />
            <img src="./Data%20Visualizations/Normalization%20(zscore_3).png" height="100%" width="30%" />
        </p>

    * **`sqrt`**

        $x' = \sqrt{max(x, 0)}$. A mild variance-stabilizing transform for nonnegative amounts; reduces skew while preserving ordering. Engineered features are mostly non-negative and skewed; thus, `sqrt` reduces skew without being as aggressive as `log`.
        <p>
            <img src="./Data%20Visualizations/Normalization%20(sqrt_1).png" height="100%" width="33%" />
            <img src="./Data%20Visualizations/Normalization%20(sqrt_2).png" height="100%" width="30%" />
            <img src="./Data%20Visualizations/Normalization%20(sqrt_3).png" height="100%" width="30%" />
        </p>

    * **`yeo_johnson`**

        A power transform $x' = T_\lambda(x)$ (works for $x \geq 0$ and $x < 0$). Learns $\lambda$ to make features more Gaussian-like without requiring positivity.
        <p>
            <img src="./Data%20Visualizations/Normalization%20(yeo_johnson_1).png" height="100%" width="33%" />
            <img src="./Data%20Visualizations/Normalization%20(yeo_johnson_2).png" height="100%" width="30%" />
            <img src="./Data%20Visualizations/Normalization%20(yeo_johnson_3).png" height="100%" width="30%" />
        </p>

    * **`quantile_normal`**

        $x' = \Phi^{-1}(F(x))$, where $F$ is the empirical CDF and $\Phi^{-1}$ is the normal inverse CDF. Maps each feature to an approximately normal distribution; a strong transform that can change relative distances.
        <p>
            <img src="./Data%20Visualizations/Normalization%20(quantile_normal_1).png" height="100%" width="33%" />
            <img src="./Data%20Visualizations/Normalization%20(quantile_normal_2).png" height="100%" width="30%" />
            <img src="./Data%20Visualizations/Normalization%20(quantile_normal_3).png" height="100%" width="30%" />
        </p>

    * **`log_then_zscore`**

        $x_1 = log(1 + x)$, then $x' = \frac{x_1 - \mu}{\sigma}$. First reduces skew, then standardizes scale so models see balanced feature ranges.
        <p>
            <img src="./Data%20Visualizations/Normalization%20(log_then_zscore_1).png" height="100%" width="33%" />
            <img src="./Data%20Visualizations/Normalization%20(log_then_zscore_2).png" height="100%" width="30%" />
            <img src="./Data%20Visualizations/Normalization%20(log_then_zscore_3).png" height="100%" width="30%" />
        </p>

    * **Summary**

        | Method          | Accuracy      | Macro F1      |
        | --------------- | ------------- | ------------- |
        | sqrt            | 0.999833      | 0.999833      |
        | quantile_normal | 0.999708      | 0.999706      |
        | log_then_zscore | 0.999667      | 0.999666      |
        | none            | 0.999667      | 0.999665      |
        | log             | 0.999667      | 0.999664      |
        | yeo_johnson     | 0.999583      | 0.999580      |
        | zscore          | 0.999542      | 0.999539      |
        | square          | 0.999167      | 0.999159      |
        | inverse         | 0.998958      | 0.998954      |

        **Normalization:** BEST = `sqrt` (acc = `0.9998` and macro_f1 = `0.9998`)

    7.2 **Regularization (best = L2)**

    * **L1 (Lasso)**

        Minimizes $L + \lambda||W||_1$. Encourages sparse weights (some go exactly to $0$), acting like built-in feature selection.

    * **L2 (Ridge)**

        Minimizes $L + \lambda||W||_2^2$. Shrinks weights to reduce overfitting and improve stability under correlated features. It also helps keep intermediate comparisons stable and reduces noise filtering in early stages.

    * **Elastic Net**

        Minimizes $L + \lambda(\alpha||w||_1 + (1 - \alpha)||w||_2^2)$. Mixes sparsity (L1) with stability (L2), often better when many features correlate.

    * **Summary**

        | Penalty    | C   | L1_Ratio | Accuracy | Macro F1 |
        | ---------- | --- | -------- | -------- | -------- |
        | l2         | 1.0 | 0.5      | 0.999833 | 0.999833 |
        | l1         | 1.0 | 0.5      | 0.999583 | 0.999577 |
        | elasticnet | 1.0 | 0.5      | 0.999542 | 0.999536 |

        **Regularization:** BEST = `l2` (params = `{'C': 1.0}`, acc = `0.9998`, and macro_f1 = `0.9998`)

    7.3 **Outlier Detection (best = `none`)**

    * **`none`**

        No outlier handling: $x' = x$. Keeps full variance, but extreme synthetic/rare months can distort scaling and model boundaries. Since the dataset is synthetic and already constraint-aware, some outliers are meaningful indicators (e.g., rare high-debt months).
        <p>
            <img src="./Data%20Visualizations/Outlier%20Method%20(none_1).png" height="100%" width="33%" />
            <img src="./Data%20Visualizations/Outlier%20Method%20(none_2).png" height="100%" width="30%" />
            <img src="./Data%20Visualizations/Outlier%20Method%20(none_3).png" height="100%" width="30%" />
        </p>

    * **`iqr`**

        Flag points outside $[Q_1 - k \cdot IQR, Q_3 + k \cdot IQR]$ where $IQR = Q_3 - Q_1$. The method is efficient for skewed data and can be used to clip extreme spending months.
        <p>
            <img src="./Data%20Visualizations/Outlier%20Method%20(iqr_1).png" height="100%" width="33%" />
            <img src="./Data%20Visualizations/Outlier%20Method%20(iqr_2).png" height="100%" width="30%" />
            <img src="./Data%20Visualizations/Outlier%20Method%20(iqr_3).png" height="100%" width="30%" />
        </p>

    * **`mad`**

        Compute $MAD = median(|x - median(x)|)$ and use a `z-score` $z = \frac{0.6745(x - median)}{MAD}$. Strongly resistant to heavy tails and works well when means/std are unreliable.
        <p>
            <img src="./Data%20Visualizations/Outlier%20Method%20(mad_1).png" height="100%" width="33%" />
            <img src="./Data%20Visualizations/Outlier%20Method%20(mad_2).png" height="100%" width="30%" />
            <img src="./Data%20Visualizations/Outlier%20Method%20(mad_3).png" height="100%" width="30%" />
        </p>

    * **Summary**

        | Outlier Method | Accuracy | Macro F1 |
        | -------------- | -------- | -------- |
        | none           | 0.999833 | 0.999833 |
        | mad            | 0.999708 | 0.999707 |
        | iqr            | 0.999625 | 0.999622 |

        **Outlier Method:** BEST = `none` (acc = `0.9998` and macro_f1 = `0.9998`)

    7.4 **Feature Selection (best = `heatmap_drop`)**

    * **Heatmap correlation**

        Uses correlation $r_{ij}$ to drop redundant features (e.g., remove one of a highly correlated pair). Helps prevent double-counting the same spending signal. Since aggregated share/rate features can be highly correlated, dropping them often improves generalization.
        <p>
            <img src="./Data%20Visualizations/Feature%20Selection%20(heatmap_correlation_1).png" height="100%" width="46%" />
            <img src="./Data%20Visualizations/Feature%20Selection%20(heatmap_correlation_2).png" height="100%" width="46%" />
        </p>

        **Heatmap-drop eval:** acc = `0.9922` and macro_f1 = `0.9923`

    * **VIF**

        $VIF_i = \frac{1}{1 - R_i^2}$, where $R_i^2$ is from regressing feature $i$ on others. Large VIF indicates multicollinearity, which can destabilize linear models and inflate variance.

        | Feature                                | VIF         |
        | -------------------------------------- | ----------- |
        | Debt_Payments__share__mean             | inf         |
        | DebtRate__mean	                     | inf         |
        | Auto_Costs__share__mean	             | 1488.229360 |
        | Healthcare_OOP__share__mean            | 1410.626246 |
        | Auto_Costs__share__median		         | 650.088079  |
        | Insurance_All__share__mean 	         | 492.828699  |
        | Healthcare_OOP__share__median 	     | 465.430529  |
        | Transportation_Variable__share__median | 404.193452  |
        | Auto_Costs__share__std	             | 290.139224  |
        | Healthcare_OOP__share__std             | 225.351986  |
        | Debt_Payments__share__std	             | 219.542359  |
        | Insurance_All__share__std	             | 139.428371  |
        | DiscretionaryRate__std	             | 139.046245  |
        | Pets__share__std	                     | 137.943165  |
        | Groceries_FoodAtHome__share__median	 | 110.582508  |

        **VIF-drop eval:** acc = `0.6399` and macro_f1 = `0.6377`

    * **Confusion-matrix entropy**

        Compute entropy $H = -\sum_kp_klogp_k$ from a model's confusion behavior (or per-feature ablations). Lower entropy typically means clearer class separation; it is used to keep features that reduce ambiguity across clusters.

        * Entropy-wrapper baseline: score = `0.3774`, f1 = `0.6377`, and ent = `1.0413` (features = `13`)
        * Entropy-wrapper final feature count: `13`

        **Entropy-selected eval:** acc = `0.6399` and macro_f1 = `0.6377`

    * **Summary**

        ![](/Data%20Visualizations/Feature%20Selection%20(summary).png)

        **Feature Selection:** BEST feature set = `heatmap_drop` (acc = `0.9922`, macro_f1 = `0.9923`, and features = `37`)

    7.5 **Scaling (best = `robust`)**

    * **`none`**

        No scaling: $x' = x$. Tree models often tolerate this, but distance/gradient-based methods (`MLP`, `logistic`) can suffer.
        <p>
            <img src="./Data%20Visualizations/Scaling%20(none_1).png" height="100%" width="33%" />
            <img src="./Data%20Visualizations/Scaling%20(none_2).png" height="100%" width="28.5%" />
            <img src="./Data%20Visualizations/Scaling%20(none_3).png" height="100%" width="28.5%" />
        </p>

    * **`zscore`**

        $x' = \frac{x - \mu}{\sigma}$. Makes features comparable in variance, improving optimization for gradient-based models.
        <p>
            <img src="./Data%20Visualizations/Scaling%20(zscore_1).png" height="100%" width="33%" />
            <img src="./Data%20Visualizations/Scaling%20(zscore_2).png" height="100%" width="28.5%" />
            <img src="./Data%20Visualizations/Scaling%20(zscore_3).png" height="100%" width="28.5%" />
        </p>

    * **`robust`**

        $x' = \frac{x - median}{IQR}$. Similar to `z-score` but resistant to outliers; good for spending data with occasional spikes. Since it is less sensitive to heavy tails than `z-score` scaling, it works well when some features have occasional spikes.
        <p>
            <img src="./Data%20Visualizations/Scaling%20(robust_1).png" height="100%" width="33%" />
            <img src="./Data%20Visualizations/Scaling%20(robust_2).png" height="100%" width="28.5%" />
            <img src="./Data%20Visualizations/Scaling%20(robust_3).png" height="100%" width="28.5%" />
        </p>

    * **`minmax`**

        $x' = \frac{x - min}{max - min}$. Forces features into $[0, 1]$; sensitive to outliers because $min / max$ can be extreme.
        <p>
            <img src="./Data%20Visualizations/Scaling%20(minmax_1).png" height="100%" width="33%" />
            <img src="./Data%20Visualizations/Scaling%20(minmax_2).png" height="100%" width="28.5%" />
            <img src="./Data%20Visualizations/Scaling%20(minmax_3).png" height="100%" width="28.5%" />
        </p>

    * **Summary**

        | Scaler | Accuracy | Macro F1 |
        | ------ | -------- | -------- |
        | robust | 0.992708 | 0.992746 |
        | zscore | 0.992083 | 0.992120 |
        | minmax | 0.975667 | 0.975786 |
        | none   | 0.943042 | 0.943272 |

        **Scaling:** BEST = `robust` (acc = `0.9927` and macro_f1 = `0.9927`)

    7.6 **Standardization (best = `none`)**

    * **`none`**

        No row-wise normalization. Preserves absolute magnitude differences between people/months. Since the data has already been normalized and scaled, there is no need for additional standardization in this case. Moreover, since the best model is tree-based, trees do not need standardized inputs.
        <p>
            <img src="./Data%20Visualizations/Standardization%20(none_1).png" height="100%" width="33%" />
            <img src="./Data%20Visualizations/Standardization%20(none_2).png" height="100%" width="28.5%" />
            <img src="./Data%20Visualizations/Standardization%20(none_3).png" height="100%" width="28.5%" />
        </p>

    * **`l2`**

        $x' = \frac{x}{||x||_2}$. Makes each row have unit Euclidean length, emphasizing composition (spend mix) over total scale.
        <p>
            <img src="./Data%20Visualizations/Standardization%20(l2_1).png" height="100%" width="33%" />
            <img src="./Data%20Visualizations/Standardization%20(l2_2).png" height="100%" width="28.5%" />
            <img src="./Data%20Visualizations/Standardization%20(l2_3).png" height="100%" width="28.5%" />
        </p>

    * **`l1`**

        $x' = \frac{x}{||x||_1}$. Makes each row sum to $1$, turning values into proportions; very interpretable for budget-share features.
        <p>
            <img src="./Data%20Visualizations/Standardization%20(l1_1).png" height="100%" width="33%" />
            <img src="./Data%20Visualizations/Standardization%20(l1_2).png" height="100%" width="28.5%" />
            <img src="./Data%20Visualizations/Standardization%20(l1_3).png" height="100%" width="28.5%" />
        </p>

    * **Summary**

        | Standardization | Accuracy | Macro F1 |
        | --------------- | -------- | -------- |
        | none            | 0.992708 | 0.992746 |
        | l2              | 0.966458 | 0.966387 |
        | l1              | 0.920958 | 0.921226 |

        **Standardization:** BEST = `none` (acc = `0.9927` and macro_f1 = `0.9927`)

    7.7 **Sampling (best = `SMOTE`)**

    * **`none`**

        Keeps the natural class distribution. Best when the training set is already balanced or needs a realistic frequency behavior.

    * **`SMOTE`**

        Creates synthetic minority examples by interpolation: $x_{new} = x + \alpha(x_{nn} - x)$. Reduces class imbalance without duplicating points, often improving boundary learning. It also reduces bias toward majority clusters and can improve decision boundaries on tabular data when minority classes are underrepresented.

    * **`SMOTEENN`**

        `SMOTE` oversampling with Edited Nearest Neighbors cleaning. Adds minority samples, then removes ambiguous/noisy points near class overlaps to sharpen boundaries.

    * **`UnderSample`**

        Randomly removes samples from the majority classes. Fast and simple, but can discard a useful variety and shrink decision coverage.

    * **Summary**

        | Sampler     | Accuracy | Macro F1 |
        | ----------- | -------- | -------- |
        | SMOTE       | 0.993000 | 0.993039 |
        | UnderSample | 0.992958 | 0.992996 |
        | none        | 0.992708 | 0.992746 |
        | SMOTEENN    | 0.979708 | 0.979812 |

        **Sampling:** BEST = `SMOTE` (acc = `0.9930` and macro_f1 = `0.9930`)

8. **Model training**

    Models trained:
    * **Logistic Regression**

        Learns linear decision boundaries with $p(y = k|x) = softmax(Wx + b)$. Strong baseline: fast, interpretable, but limited when clusters are non-linear.

        **LR results:** acc = `0.9930` and macro_f1 = `0.9930`

        |              | Precision | Recall  | F1-Score | Support  |
        | ------------ | --------- | ------- | -------- | -------- |
        | 0            | 0.998     | 0.997   | 0.997    | 4082     |
        | 1            | 0.985     | 0.989   | 0.987    | 4010     |
        | 2            | 0.989     | 0.986   | 0.987    | 4109     |
        | 3            | 0.994     | 0.995   | 0.994    | 3939     |
        | 4            | 0.993     | 0.996   | 0.995    | 3958     |
        | 5            | 0.999     | 0.995   | 0.997    | 3902     |
        |              |           |         |          |          |
        | accuracy     |           |         | 0.993    | 24000    |
        | macro avg    | 0.993     | 0.993   | 0.993    | 24000    |
        | weighted avg | 0.993     | 0.993   | 0.993    | 24000    |

        ![](/Data%20Visualizations/Model%20(logistic_regression).png)

    * **Random Forest**

        Ensemble of bootstrapped decision trees with feature randomness. Captures non-linear interactions, but can be less sharp than boosted methods.

        **RandomForest results:** acc = `0.9956` and macro_f1 = `0.9956`

        |              | Precision | Recall  | F1-Score | Support  |
        | ------------ | --------- | ------- | -------- | -------- |
        | 0            | 0.998     | 0.999   | 0.998    | 4082     |
        | 1            | 0.988     | 0.997   | 0.992    | 4010     |
        | 2            | 0.995     | 0.989   | 0.992    | 4109     |
        | 3            | 0.996     | 0.996   | 0.996    | 3939     |
        | 4            | 0.997     | 0.996   | 0.997    | 3958     |
        | 5            | 1.000     | 0.997   | 0.998    | 3902     |
        |              |           |         |          |          |
        | accuracy     |           |         | 0.996    | 24000    |
        | macro avg    | 0.996     | 0.996   | 0.996    | 24000    |
        | weighted avg | 0.996     | 0.996   | 0.996    | 24000    |

        ![](/Data%20Visualizations/Model%20(RandomForest).png)

    * **ExtraTrees**

        Like Random Forest, but splits are more randomized. Often reduces variance and trains faster, sometimes at the cost of slightly higher bias.

        **ExtraTrees results:** acc = `0.9952` and macro_f1 = `0.9952`

        |              | Precision | Recall  | F1-Score | Support  |
        | ------------ | --------- | ------- | -------- | -------- |
        | 0            | 0.999     | 0.993   | 0.996    | 4082     |
        | 1            | 0.984     | 0.995   | 0.989    | 4010     |
        | 2            | 0.995     | 0.990   | 0.993    | 4109     |
        | 3            | 0.997     | 0.998   | 0.998    | 3939     |
        | 4            | 0.997     | 0.998   | 0.997    | 3958     |
        | 5            | 1.000     | 0.997   | 0.998    | 3902     |
        |              |           |         |          |          |
        | accuracy     |           |         | 0.995    | 24000    |
        | macro avg    | 0.995     | 0.995   | 0.995    | 24000    |
        | weighted avg | 0.995     | 0.995   | 0.995    | 24000    |

        ![](/Data%20Visualizations/Model%20(ExtraTrees).png)

    * **XGBoost (BEST)**

        Gradient-boosted trees: $F_t(x) = F_{t - 1}(x) + \eta f_t(x)$. Typically excels on tabular data by iteratively correcting errors and modeling complex feature interactions. This model performs well on tabular and mixed-signal feature sets. Thus, it will capture non-linear thresholds (e.g., a high savings rate with a low housing share) and perform well without requiring deep feature scaling.

        **XGBoost results:** acc = `0.9960` and macro_f1 = `0.9961`

        |              | Precision | Recall  | F1-Score | Support  |
        | ------------ | --------- | ------- | -------- | -------- |
        | 0            | 0.998     | 0.998   | 0.998    | 4082     |
        | 1            | 0.989     | 0.995   | 0.992    | 4010     |
        | 2            | 0.996     | 0.991   | 0.993    | 4109     |
        | 3            | 0.997     | 0.997   | 0.997    | 3939     |
        | 4            | 0.997     | 0.997   | 0.997    | 3958     |
        | 5            | 1.000     | 0.997   | 0.999    | 3902     |
        |              |           |         |          |          |
        | accuracy     |           |         | 0.996    | 24000    |
        | macro avg    | 0.996     | 0.996   | 0.996    | 24000    |
        | weighted avg | 0.996     | 0.996   | 0.996    | 24000    |

        ![](/Data%20Visualizations/Model%20(XGBoost).png)

    * **MLP**

        A feed-forward neural network trained by backpropagation. It can fit complex patterns but needs careful scaling/regularization and is more sensitive to the dataset than tree boosting.

        **MLP results:** acc = `0.9960` and macro_f1 = `0.9960`

        |              | Precision | Recall  | F1-Score | Support  |
        | ------------ | --------- | ------- | -------- | -------- |
        | 0            | 0.998     | 0.998   | 0.998    | 4082     |
        | 1            | 0.989     | 0.995   | 0.992    | 4010     |
        | 2            | 0.994     | 0.990   | 0.992    | 4109     |
        | 3            | 0.998     | 0.996   | 0.997    | 3939     |
        | 4            | 0.997     | 0.998   | 0.997    | 3958     |
        | 5            | 0.999     | 0.998   | 0.999    | 3902     |
        |              |           |         |          |          |
        | accuracy     |           |         | 0.996    | 24000    |
        | macro avg    | 0.996     | 0.996   | 0.996    | 24000    |
        | weighted avg | 0.996     | 0.996   | 0.996    | 24000    |

        ![](/Data%20Visualizations/Model%20(MLP).png)

    * **Summary**

        | Model        | Accuracy | Macro F1 |
        | ------------ | -------- | -------- |
        | xgboost      | 0.996042 | 0.996066 |
        | mlp          | 0.995958 | 0.995986 |
        | randomforest | 0.995583 | 0.995608 |
        | extratrees   | 0.995167 | 0.995207 |
        | lr           | 0.993000 | 0.993039 |

        <p>
            <img src="./Data%20Visualizations/Model%20(best_1).png" height="100%" width="46%" />
            <img src="./Data%20Visualizations/Model%20(best_2).png" height="100%" width="46%" />
        </p>

        **Model:** BEST MODEL = `xgboost` (acc = `0.9960` and macro_f1 = `0.9961`)

9. **Saving artifacts**

    Once the best pipeline configuration is selected, the notebook writes everything needed for consistent inference:

    * `best_cluster_model.pkl` - trained classifier (`XGBoost` used as a final model in production)
    * `feature_cols.json` - feature column order expected by the model
    * `pipeline_config.json` - selected preprocessing choices with key parameters (`norm` / `scaling` / `selection` / etc.)
    * `outlier_bounds.pkl` - per-feature bounds used by the outlier step (`IQR` / `MAD`),  or identity if none
    * `norm_obj.pkl` - normalization transform object, or identity if none
    * `standardization_obj.pkl` - `L1` / `L2` / `none` standardizer,  or identity if none
    * `scaler_obj.pkl` - scaling transform (`robust`/`minmax`/`zscore`/`none`)
    * `sampler_obj.pkl` - the resampling strategy used during training (`SMOTE`, `SMOTEENN`, `UnderSample`, or identity for none)

    These artifacts are saved for the backend (`predict_api.py`).

10. **Cluster space clouds**

    After training, a cloud of synthetic probe feature vectors was run through the same transforms to predict cluster labels and project to 2D for plotting (PCA in the original version).

    This produces cached visualization files like:
    * `cluster_space_cache.joblib`
    * `cluster_space_cache.json`

    Those files are used by the frontend to show cluster regions without recomputing projections every time. The cluster space clouds in the frontend do not look like clusters right after training, as:

    * It is a 2D PCA projection of a high-dimensional space; thus, even if clusters are separable in $10$–$50$ dimensions, PCA(2) can squash that separation as PCA preserves variance, not class separation.
    * Cloud points are synthetic and a cache builder samples features broadly, often uniformly within bounds.
    * The model's boundaries are likely nonlinear; therefore, since the classifier is tree-based (`XGBoost`), the decision regions can be complex in high-dimensional space.
    * The clusters are income tiers, meanwhile the features are ratios/shares. Ratios (shares of income) often overlap across tiers. For example, someone at $3,000/mo and $6,000/mo can have very similar shares, so the features can legitimately overlap.

    ![](/Data%20Visualizations/Model%20(cluster_space).png)

11. **Verification & Testing**

    Finally, the notebook (and `model_test.ipynb`) includes behavioral tests that go beyond standard train/test metrics:

    * **One-month test:** feeds a single month and verifies that the model still produces a reasonable cluster distribution with warnings that single-month inputs are inherently less stable.

    * **6-month behavior timeline:** simulates short sequences to confirm the prediction responds to sustained behavior rather than one anomaly.

        Table 1: Single-month prediction with rates

        | month | SavingsRate | NetCashflowRate | pred_single_month | top_prob_% | p_C1_low | p_C2_lower_mid | p_C3_mid | p_C4_upper_mid | p_C5_high | p_C6_top5 |
        | :---: | :---------: | :-------------: | :---------------: | :--------: | :------: | :------------: | :------: | :------------: | :-------: | :-------: |
        |   1   |   0.133333  |     0.143333    |    C2_lower_mid   |  81.103020 | 0.001928 |    0.811030    | 0.014249 |    0.056035    |  0.008090 |  0.108669 |
        |   2   |   0.150000  |     0.131667    |    C2_lower_mid   |  81.552383 | 0.002057 |    0.815524    | 0.013991 |    0.056058    |  0.008216 |  0.104154 |
        |   3   |   0.166667  |     0.120000    |    C2_lower_mid   |  81.916595 | 0.002104 |    0.819166    | 0.013936 |    0.055986    |  0.008447 |  0.100361 |
        |   4   |   0.183333  |     0.108333    |    C2_lower_mid   |  81.526527 | 0.002661 |    0.815265    | 0.013253 |    0.061350    |  0.008343 |  0.099128 |
        |   5   |   0.200000  |     0.096667    |    C2_lower_mid   |  82.131279 | 0.004506 |    0.821313    | 0.013425 |    0.048137    |  0.008577 |  0.104041 |
        |   6   |   0.216667  |     0.085000    |    C2_lower_mid   |  82.481873 | 0.005593 |    0.824819    | 0.010473 |    0.047646    |  0.008490 |  0.102980 |

        Table 2: Timeline prediction (cumulative months 1..t)

        | months_used | pred_index |   pred_name  | top_prob_% | p_C1_low | p_C2_lower_mid | p_C3_mid | p_C4_upper_mid | p_C5_high | p_C6_top5 |
        | :---------: | :--------: | :----------: | :--------: | :------: | :------------: | :------: | :------------: | :-------: | :-------: |
        |      1      |      1     | C2_lower_mid |  81.103020 | 0.001928 |    0.811030    | 0.014249 |    0.056035    |  0.008090 |  0.108669 |
        |      2      |      1     | C2_lower_mid |  81.433777 | 0.002054 |    0.814338    | 0.013971 |    0.055976    |  0.008092 |  0.105569 |
        |      3      |      1     | C2_lower_mid |  81.552383 | 0.002057 |    0.815524    | 0.013991 |    0.056058    |  0.008216 |  0.104154 |
        |      4      |      1     | C2_lower_mid |  82.005539 | 0.002031 |    0.820055    | 0.014038 |    0.055360    |  0.008268 |  0.100247 |
        |      5      |      1     | C2_lower_mid |  81.916595 | 0.002104 |    0.819166    | 0.013936 |    0.055986    |  0.008447 |  0.100361 |
        |      6      |      1     | C2_lower_mid |  81.541512 | 0.002217 |    0.815415    | 0.013516 |    0.061361    |  0.008344 |  0.099146 |

        ![](/Data%20Visualizations/Model%20(6_month_test).png)

    * **48-month what-if transition (C1 → C4):** use a heuristic blending function that gradually morphs spending shares from one cluster's profile to another and confirms the model transitions across clusters over time instead of snapping randomly.

        ![](/Data%20Visualizations/Model%20(48_month_test).png)
        ![](/Data%20Visualizations/Model%20(48_month_test_radar).png)

    These tests are vital, since the app is meant to behave like a personal dynamic financial trajectory estimation tool rather than just a static classifier.

## Front end

### `FinGrowthDashboard.js` (dashboard UI)

**Purpose:** One-page dashboard that collects monthly totals for 17 categories, calls the backend, and renders results.
    <p>
        <img src="./Data%20Visualizations/UI%20(dashboard_1).png" height="100%" width="46%" />
        <img src="./Data%20Visualizations/UI%20(dashboard_2).png" height="100%" width="46%" />
    </p>

**UI layout:**

* 4 radar charts (`Snapshot` / `Trend` / `Risk` / `Growth`)

    ![](/Data%20Visualizations/UI%20(radars).png)

    * **`Snapshot`**

        Shows the latest month profile as normalized behavior scores (`Essentials`, `Debt`, `Savings`, `Fun`, and `Left`). It helps understand the current state without needing a long history. It updates only after Save, so the chart always matches the same model-backed run.

    * **`Trend`**

        Aggregates the recent months' data (rolling average) to show the user's typical behavior over the past few months. This smooths out one-off months and highlights consistent patterns. It is useful when entering multiple months to see the recent trend.

    * **`Risk`**

        Represents a conservative stability score across all provided months (penalizes volatility). If the user's spending behavior is inconsistent, `Risk` drops even if Snapshot looks fine. This panel is designed to reflect safe/steady patterns over time.

    * **`Growth`**

        Measures momentum by comparing the most recent window vs the previous window ($50$ = no change). Values above $50$ indicate improving signals (e.g., a higher `Savings` score), while values below $50$ indicate a decline. In other words, it shows the trend of progress.

* Net worth line chart with cluster-space scatter

    ![](/Data%20Visualizations/UI%20(net_and_cluster).png)

    * **`Net worth`**

        Plots a net worth / savings trajectory over time, starting from a baseline and projecting forward. It uses the user's income and outflows to estimate month-to-month net changes, then extrapolates a short forecast.

    * **`Cluster space`**

        Visualizes all 6 clusters as a 2D cloud (PCA projection) and places "You" (user) as a highlighted point. This shows where user's features land relative to the cluster regions, not just the final label.

* **Monthly input table** (add/delete months with autofill)

    This is the data entry surface: add $1+$ months of totals across $17$ categories. More months generally improve the stability of features like `mean`/`std`, making `Trend`/`Risk`/`Growth` more meaningful. Autofill can generate realistic sample months per cluster to demo the system without manual typing. Adding or deleting a month updates the month counter in the top bar, and clicking Save calls `onSave()`, refreshes the "Updated HH:MM:SS" status timestamp in the top bar, and re-computes all charts/panels from the latest API response.

    ![](/Data%20Visualizations/UI%20(monthly_inputs).png)

* **Conclusion panel** (probabilities, drivers, missing fields, warnings, and errors)

    Summarizes the top predicted cluster plus the full probability distribution across all clusters. It also lists top drivers (largest spend shares relative to income) and missing fields that may reduce accuracy. This panel is designed to be the single place a user reads when they want the answer fast.

    ![](/Data%20Visualizations/UI%20(conclusion_1).png)

* **Warnings and errors**

    Warnings are non-fatal issues detected during parsing or inference (e.g., missing income, non-finite values, missing optional caches). They do not stop prediction, but they tell a user why results may be less reliable or why some visuals (like cluster space) might be empty. Errors occur when the API request fails (server is offline, CORS/network issues, or a malformed response), and the UI displays a red error box while keeping the rest of the dashboard intact. Together, warnings and errors prevent silent failures and make the app safer to scale: the frontend avoids repeated requests, and the backend can communicate degraded mode.

    ![](/Data%20Visualizations/UI%20(conclusion_2).png)

**State flow:**
`rows` → POST to `/predict` → store `clusterResult` with `analysis` → charts rerender from backend response only.

**Autofill:** Generates realistic month examples by cluster using deterministic RNG (`mulberry32`) so the same selections look stable across refreshes.

**Backend string cleanup:** `FIELD_LABEL_MAP` and `prettifyBackendLine()` converts backend keys like `Groceries_FoodAtHome` into human labels (`Groceries`) for drivers/warnings.

**Important detail:** the frontend normalizes response shape (radars, cluster points, warnings) so the UI does not crash when optional fields are missing.

### Predict API call (frontend)

* `PREDICT_API_URL = http://127.0.0.1:5055/predict`
* `onSave()` does:
    * `onSave()` sets an `apiBusy` flag before sending the request, and the Save button is disabled while `apiBusy=true`. This blocks UI updates / double-submits while the model pipeline is running, avoiding overlapping requests or race condition responses.
    * If the server is down (or returns non-200), the call fails fast, `apiError` is populated, and the UI stays stable instead of retry-spamming the backend. This pattern scales well later (multi-user / multi-worker servers).
    * `fetch(..., { method:"POST", body:{ months: rows } })`
    * validates response has `{ top, probs }`
    * stores `analysis` payload used by charts and conclusion

## Back end

### `predict_api.py` (FastAPI inference server)

**Purpose:** Turns `{ months: [...] }` into:
* `top` cluster and `probs`
* `radars` (4 radar datasets)
* `networth` forecast series
* `cluster_space` cloud and user projection
* `conclusion` text, drivers, and missing fields
* `warnings` (data and artifact warnings)

### How the server works

* **Load artifacts** from `artifacts/` at startup (model, preprocessing objects, and config)

* **`/predict`:**

    * **Parse input months** → **DataFrame** and detect missing fields and warnings on bad income
    * **Feature engineering** (single-row feature vector)
    * **Inference transforms** (must match training order):
        * norm → outlier_clip → scaler → standardization
    * **Predict probabilities** (`predict_proba` → fallback to softmax/onehot)
    * **Generate charts** (radars and net worth)
    * **Cluster space:** project user point using cached PCA params and return sampled cloud
    * **Build conclusion:** top drivers based on latest-month shares

* **`/health`** (check endpoint)
Returns which artifacts exist with a pipeline config and warnings (e.g., missing sampler/cache).

## Feature engineering (`feature_engineering.py`)

* **Purpose:** Convert raw monthly dollars into model-behavior features.
* **Main idea:** Use shares/rates so users are comparable across incomes:
    * category_share = `category / Income_Deposits`
    * NetCashflowRate = `(Income - sum(outflows)) / Income`
    * EssentialRate = `(Housing + Utilities + Groceries) / Income`
    * DiscretionaryRate = `(Dining + Entertainment + Travel + Subscriptions) / Income`
* **Across-month aggregation:** compute `mean`/`median`/`std` ($ddof = 0$) for selected shares/rates → output `feature_cols.json`.
* **Robustness:** `_normalize_legacy_columns()` maps older keys into the merged 17-category schema so the model will not silently get zeros.

## Routing (`AppRoutes.js`)

* **Purpose:** Connects pages with React Router
    * `/` → `ReactRoot` (canvas playground)
    * `/fin-growth-dashboard` → `FinGrowthDashboard` (finance UI)

## Canvas entry with styles

### `index.js` (canvas playground)

* **Purpose:** A separate interactive canvas demo that also includes a button to navigate to FinGrowth.
* **Key parts:**
    * Uses `useNavigate()`  to route to `/fin-growth-dashboard`
    * Renders a `<canvas>` and a "turtle" marker whose position/angle are mirrored in React state
    * Mouse movement updates `lastCursor` (used as origin for fractals)
    * Implements turtle drawing with animated Koch snowflake and Hilbert curve using stack-based expansion (avoids recursion and supports smooth animation)

### `styles.js`

* **Purpose:** JS style file for the canvas UI:
    * full-screen fixed layout (`root`)
    * gradient header
    * reusable button styles, canvas wrapper, and a turtle triangle marker

## Running the backend locally (Windows)

### `requirements_api.txt`

Pinned versions for FastAPI, uvicorn, and ML stack.

### `run_api.bat`

Runs:
```uvicorn predict_api:app --host 127.0.0.1 --port 5055 --reload```