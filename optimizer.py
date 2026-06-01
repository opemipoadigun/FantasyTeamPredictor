# 1. Import libraries (pandas, pulp)
import pandas as pd
from pulp import *

# 2. Load the predictions from predict.py output
#    (you'll need to save predictions to a CSV first — add that to predict.py)
df = pd.read_csv("data/processed/predictions.csv")
# 3. Set up the constraints (budget, position counts, max per team)

# 4. Create a PuLP problem and decision variables
#    (one variable per player — 1 if selected, 0 if not)

# 5. Add the objective — maximise total predicted points

# 6. Add the constraints

# 7. Solve it

# 8. Print the selected squad