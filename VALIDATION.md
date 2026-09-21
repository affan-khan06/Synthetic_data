# Validation performed

Validated on Linux with Python 3.12.14. Windows BAT launchers were inspected, not executed on Windows.

Passed:
- Actual UCI workbook load: 30,000 rows, 25 columns, 6,636 default outcomes.
- Full default VAE/diffusion training on real training records and measured validation evaluation.
- Separate frozen-configuration evaluation against real test records.
- Five tests: source/schema checks, no split overlap, signed-number transform round-trip, fresh generation validity, malformed-source rejection.
- Streamlit AppTest: all seven tabs load with no exception; fresh generation creates 1,000 profiles; real-data quick training and restore both succeed.
- Package CSV outputs are generated from the trained model; no simulated source fallback exists.

Browser screenshot verification was not performed. The previous browser runtime was unavailable. AppTest checks execution, not visual rendering. Numerical results may vary across library versions/hardware.

Versions used:

{
  "torch": "2.14.0",
  "numpy": "2.3.5",
  "pandas": "2.2.3",
  "scipy": "1.17.0",
  "scikit-learn": "1.8.0",
  "streamlit": "1.64.0",
  "plotly": "7.1.0",
  "xlrd": "2.0.1"
}