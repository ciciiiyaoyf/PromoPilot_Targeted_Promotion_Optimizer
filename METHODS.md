# How PromoPilot works

## 1. The decision and its objective

The business question is: **which offer, if any, should each customer receive
to increase total 14-day net contribution margin?** A customer who is very
likely to buy may buy without a discount. Purchase probability alone therefore
does not identify the most profitable promotion recipients.

For customer `i` and action `a`, the profit model predicts `NCM(i, a)`. The value
used by the allocator is:

```text
incremental_NCM(i, a) = predicted_NCM(i, a) - predicted_NCM(i, NO_OFFER)
```

The four actions are `NO_OFFER`, `PCT_10`, `PCT_20`, and `FREE_SHIP`. The model
predicts all four possibilities, although the historical data reveal only the
outcome under the action actually assigned. These counterfactual predictions
are estimates, not four observed outcomes for each customer.

## 2. Clean data and construct the outcome

Each historical modeling row represents one customer-campaign assignment.
The pipeline validates the raw tables before joining them. The data-quality
registry includes considerably more than five substantive checks: unique
identifiers, valid foreign keys, timestamp consistency, order/item revenue and
cost reconciliation, cumulative refund limits, correct discount/redemption
rules, valid campaign attribution, and nonoverlapping outcome windows.
Invalid input stops the pipeline; permitted cleanup is explicitly reported.

- **Duplicate raw records:** remove identical full rows once and record the
  count. Different rows sharing a primary key fail validation. A customer may
  have only one assignment within a campaign.
- **Orders:** count completed orders linked to the assignment's customer and
  campaign, with `send_ts <= order_ts < send_ts + 14 days`. Cancelled orders
  do not create purchases or margin. Organic preperiod orders inform features
  but do not become campaign outcomes. Do not attach an order to a nearby
  campaign merely because its date is convenient.
- **Returns:** sum distinct partial refunds without duplicating their original
  order or items. Subtract refunds recorded inside the order's campaign outcome
  window from the primary target. Later refunds are disclosed separately as a
  retrospective sensitivity, not used as labels available at an earlier fit.
- **Multiple campaign exposures:** retain separate assignment rows across
  campaigns. Historical feature aggregation uses only earlier information.
  Overlapping historical campaign windows are rejected because this dataset's
  attribution design assumes disjoint windows. Evaluation uncertainty is
  clustered by customer to account for repeated observations.

The binary target is `purchase_14d = 1` when at least one eligible completed
order exists. The signed profit target is:

```text
NCM = net_revenue - cost_of_goods - shipping_cost - applicable_refunds
net_revenue = merchandise_gross_revenue - discount_amount + shipping_revenue
```

Discounts are already reflected in net revenue and are **not subtracted again**.
A nonbuyer has zero campaign NCM; a buyer can have negative NCM after costs and
refunds. Promotion cost is a separate budget quantity: redeemed discounts plus
inferred customer shipping fees waived by a redeemed free-shipping offer.

The agreed shipping assumption is a **$75 merchandise basket cutoff**: the
waived fee is $5.99 below $75 and zero at or above $75. The supplied records
support $5.99 as a standard charged fee, but do not explicitly provide a complete
shipping tariff. This is an assumption about forgone customer shipping revenue,
not a claim that the retailer's fulfillment cost is $5.99. The cutoff and fee
are explicit configuration values.

## 3. Use only information available at the decision

Historical features use events **strictly before `send_ts`**. July features use
events **at or before each customer's supplied `policy_as_of_ts`**. An outcome
may be used for training only once its full 14-day window has matured by the
next decision date.

For the supplied July scoring population, the June 20, 2025 policy cutoff is
the information boundary. "Frozen" means later orders, sessions, email events,
and returns cannot change those scoring features. The campaign send date is a
different field. There is no artificial rule that historical features must
stop seven days before their send dates.

The original extracts contain limited transaction coverage. The last ordinary
preperiod order is January 9, 2025; campaign-attributed orders are a different
source. A latest observed timestamp does not certify that every purchase up to
that date was captured, or that subsequent customers stopped buying.

Features such as `observed_orders_30d` count completed orders actually present
in the extract within the specified window. **An observed zero is not a claim
of zero real purchases.** `transaction_coverage.csv` separately reports those
observations and leaves complete-history counts unknown unless an explicit
all-source coverage contract, available at the decision time, certifies the
entire window. No such contract is supplied to the current run, so those
complete-history counts remain unknown. Coverage diagnostics do not silently
rewrite the 43 model predictors or shift their dates.

Customer profile attributes come from a single supplied snapshot. Because the
extract has no profile change history, historical availability of changing
attributes such as membership or tier cannot be independently reconstructed.
This is an input limitation even when event timestamp checks pass.

## 4. The models

### One purchase-propensity baseline

The baseline is the retained original logistic regression, with `C=0.01`,
numeric median imputation, missing-value indicators, numeric scaling, and
categorical one-hot encoding. All 43 prespecified predictors are used: profile
(8), observed purchases (13), observed returns (3), web activity (8), earlier
email activity (7), and observed product history (4). Encoding and missingness
indicators expand the number of model columns beyond 43.

There is no separate "selected logistic" and no automatic feature-selection
step. Regularization shrinks coefficients. `offer_type`, current-campaign
responses, and outcome columns are excluded from the baseline predictors.
It ranks customers by propensity to purchase under the historical assignment
mix, using all eligible historical assignments. It does not estimate untreated
purchase probability or incremental promotion response.

For a feasible benchmark policy, the highest-propensity eligible customers
receive the same 10% offer, subject to the shared caps and budget. Holding the
offer fixed makes this a customer-targeting benchmark rather than a second
profit optimizer.

### Offer-conditioned signed-profit model

The deployed profit model is a two-part XGBoost model:

```text
expected_NCM(action) = P(purchase | features, action)
                      * [0.5 * predicted_buyer_NCM(features, action)
                         + 0.5 * mean_buyer_NCM_for_action]
```

The purchase model learns from buyers and nonbuyers. The margin model learns
from purchasers and keeps negative profit values. The 50% shrinkage toward each
action's purchaser mean reduces sensitivity to noisy individual margin
predictions. This decomposition handles the many zero outcomes explicitly;
it is not a guarantee of better accuracy or better policy value.

Both components share information across actions and include an internal action
feature. They train on `exploration` assignments; mixing in the deliberately
targeted business-rule assignments would undermine the randomized comparison.
The purchase component uses 150 trees, depth 2, and learning rate 0.03; the
margin component uses 100 trees, depth 2, and learning rate 0.03. Other fixed
regularization settings are visible in `models.py`. Randomness uses seed 42.

Promotion costs are learned separately. For each commercial action, an XGBoost
cost predictor is compared with the action's average cost using CP06, fitted
on earlier campaigns for that comparison. The chosen family is then refit on
eligible history. The retained choices are XGBoost for 10% and 20% discounts
and the action mean for free shipping. No-offer promotion cost is zero. The
reference cost configuration is 100 trees, depth 1, learning rate 0.03.

The same fixed reference regressor configuration is also used for four
action-specific nuisance outcome models in historical doubly robust evaluation.
These are evaluation components, not additional deployment candidates.

## 5. Evaluation and what the numbers mean

Rolling development fits on earlier campaigns and predicts the next campaign.
CP04-CP06 predictions determine the logistic classification threshold by maximum
pooled F1; that threshold is then frozen. This preserves chronology and avoids
letting the evaluation campaign choose its own threshold. It does not require
repeatedly searching model families or changing the historical feature window.

CP07 is the final historical evaluation campaign. It was already inspected
during project development, so its results are **retrospective evidence**, not an
untouched prospective test of all prior model decisions. Final July fits use
the eligible historical campaigns, with no July outcomes.

- **Average precision (AP)** summarizes purchase ranking across thresholds.
  **Recall** is the share of actual buyers identified at the frozen threshold;
  precision reveals how many flagged customers actually buy. Log loss and
  Brier score assess probability quality. A threshold change can trade recall
  against precision without improving ranking.
- **RMSE** measures the typical scale of dollar prediction error with extra
  weight on large errors. **MAE** measures average absolute dollar error.
  Additional regression summaries describe bias and fit. These assess NCM
  under observed actions; they do not establish that a predicted incremental
  offer effect is correct. Per-row percentage errors such as MAPE are unsuitable
  when actual NCM can be zero or negative. The reported aggregate WAPE instead
  divides total absolute error by total absolute actual NCM; it is undefined
  if every actual outcome is zero.
- **Historical policy evaluation** uses exploration assignments and contrasts
  each policy with no offer using doubly robust estimation. Nuisance models
  are fitted on earlier campaigns. Customer-cluster bootstrap intervals expose
  sampling uncertainty. Randomized assignment and adequate action support are
  assumptions required to interpret these as causal comparisons.
- **July projections** sum the fitted models' values under the chosen actions.
  They are planning estimates, not measured campaign uplift or guaranteed profit.

The verified run recorded logistic AP
**0.232325** and recall **0.608782**, and profit RMSE **$62.73847** and MAE
**$26.48059**. Its July policy projected **$2,501.7834 incremental NCM** with
**$167.44344 promotion cost**, assigning 1,665 customers free shipping and 1,111
no offer, with no percentage discounts. Historical doubly robust incremental NCM was **$8.88876
per customer**, with a customer-bootstrap 95% interval of **$0.25282 to
$18.02990**. That interval conditions on the fitted models and policy, and does
not capture future population drift or all uncertainty from earlier model
selection. The generated report records the results of each subsequent run.

## 6. Allocation is separate from XGBoost

The allocator is `scipy.optimize.milp`, using HiGHS. It chooses one binary
customer-action assignment per customer to maximize the sum of predicted
incremental NCM, with these constraints:

- Total predicted promotion cost is at most **$62,000**.
- At most **60% of all scoring customers** receive a commercial offer.
- `PCT_20` recipients are at most **15% of all commercial-offer recipients**.
- The reserved measurement controls receive `NO_OFFER`.

For 2,776 customers, the offer cap is `floor(0.60 * 2776) = 1665`. Every
customer can receive no offer; there is no minimum quota for any discount.
The solver first maximizes NCM, then breaks near-ties using fewer offers, cost,
and stable ordering while preserving the first objective within numerical
tolerance. Its optimality certificate concerns these predicted values and
constraints, not the accuracy of the models. A different optimizer cannot
improve an already optimal objective without changing those inputs.

The retained solution has no percentage discounts because they do not win
under the current predictions and constraints. That is a reason to examine
model validity and uncertainty, not a mathematical requirement to assign all
offer types.

## 7. Measurement controls and the two proposed alternatives

The current implementation reserves `ceil(0.05 * 2776) = 139` customers by
simple random sampling with seed 42 before allocation. The other 2,637 form
the policy arm. Some policy-arm customers also receive no offer because of
the optimization. They remain in the policy arm for measurement; do not move
them into the randomized control arm after seeing their assignment.

The intended experiment compares average outcomes for the complete policy arm
against the reserved control arm. Comparing only offer recipients against
controls would reintroduce selection by the policy. Five percent controls can
also produce wide uncertainty intervals; randomization does not guarantee a
precise estimate.

The following are **options for the project manager**, not implemented changes:

| Approach | How it chooses the 5% | Advantages | Limitations |
| --- | --- | --- | --- |
| **Stratified random selection** | Define groups from information known before assignment, such as predicted-gain quantiles, then randomize controls within every group. Prespecify boundaries, sample sizes, and seed. | Gives representation across customer types and supports defensible measurement. Can improve balance and precision when groups predict outcomes. | Still withholds offers from some promising customers. Proportional 5% sampling does not automatically reduce foregone NCM. Unequal sampling rates require suitable weighting and may increase uncertainty. |
| **Optimize first; select controls from assigned `NO_OFFER` customers** | Optimize without reserved controls, then designate 139 customers among those already assigned no offer. | Can avoid extra projected opportunity cost from withholding offers, if enough no-offer customers exist. | The controls are selected for their predicted response, so they are not comparable to the policy population. A simple outcome difference cannot measure campaign-wide causal uplift; model-based adjustment adds assumptions and does not restore randomization. |

No lower-/higher-gain cutoff is hard-coded for the proposed stratified design.
It must be specified before drawing controls; quantiles are one defensible
choice. Neither approach guarantees a particular extra dollar return. Choose
between them based on whether credible campaign-wide measurement or avoiding
projected control opportunity cost is the primary requirement.
