# Dataset dictionary

Source: Yeh, I. (2009). Default of Credit Card Clients. UCI. https://doi.org/10.24432/C55S3H. CC BY 4.0. Original columns and values are preserved except target renaming to `default` and column reordering.

| Field | Meaning |
|---|---|
| ID | Original source row/client identifier. Excluded from all modeling. |
| synthetic_id | Assigned to exported generated records only. No correspondence to source ID. |
| LIMIT_BAL | Given credit limit, NT$ |
| AGE | Age in years |
| SEX | Source code: 1 male, 2 female |
| EDUCATION | 1 graduate school, 2 university, 3 high school, 4 other; observed 0/5/6 retained as undocumented codes |
| MARRIAGE | 1 married, 2 single, 3 other; observed 0 retained as undocumented |
| PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6 | Repayment status for September, August, July, June, May, April 2005 respectively. -1: paid duly; 1–8: months delayed; 9: nine or more months delayed. Source -2 and 0 are retained without assigning meanings not given in the UCI summary. |
| BILL_AMT1 … BILL_AMT6 | Bill amounts for September backwards through April 2005, NT$. Negative source amounts are retained. |
| PAY_AMT1 … PAY_AMT6 | Previous payment amounts for September backwards through April 2005, NT$ |
| default | Original `default payment next month`: 1 yes, 0 no |

Source CSV: 25 columns (ID + 23 features + target). Synthetic export: 25 columns (synthetic_id + same 23 features + target). No names, addresses, salaries, villages or occupations are present. Age and source category fields are modeled jointly, not randomly appended after learning.

No missing values are filled. All 30,000 rows are retained in the normalized source CSV. Before fitting, 35 identical feature+target rows are removed and the rest split with fixed seeds. The original XLS and hashes are included for traceability. An anonymized public source and zero generated duplicates do not imply formal privacy protection.
