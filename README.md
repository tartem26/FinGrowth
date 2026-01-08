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
