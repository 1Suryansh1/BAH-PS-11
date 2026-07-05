def generate_results_table():
    print("Generating Results Table...")
    table = """
BEN-14K Results (F1@5):
  Method        Params(M)  S1->S1   S2->S2   S1->S2   S2->S1
  -----------   ---------  -----   -----   -----   -----
  MAE            224.87     60.81   72.04   41.78   46.12
  SatMAE         329.40     70.86   78.71   49.57   52.48
  DeCUR          250.54     71.26   75.36   40.78   41.83
  REJEPA         197.09     76.38   75.42   55.46   56.32
  X-JEPA         172.86     72.98   82.65   61.23   63.73
  CR-JEPA        117.93     75.11   82.87   75.82   75.40
  Ours (CopFM-R) ~15M*      TBD     TBD     TBD     TBD
    """
    print(table)
    
if __name__ == "__main__":
    generate_results_table()
