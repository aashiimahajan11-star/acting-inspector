# All costs in EUR per part. These are ILLUSTRATIVE ASSUMPTIONS:
# change them to match the scenario you want to model.
COSTS = {
    "miss": 10_000,      # defective part ships to the customer (returns, warranty, recall share)
    "false_stop": 200,   # good part triggers an automatic stop (lost output + engineer check)
    "review": 4,         # a person inspects one flagged part (about 2 minutes)
    "catch": 0,          # defect correctly removed (it would be scrapped anyway)
}

HUMAN_ACCURACY = 0.98    # share of defects a reviewer correctly confirms
DEFECT_RATE = 0.02       # realistic share of defective parts on the line (2%)

# --- Step 7: actions and alerts ---
ALERTS_ENABLED = True
ALERT_TOPIC = "acting-inspector-387a1653"     # your private ntfy topic
PAUSE_SECONDS = 30         # how long an automatic line pause lasts
