
from pathlib import Path

ROOT = Path(".")

required = [
    "utils_eval.py",
    "get_main.py",
    "main_RandomSearch.py",
    "summarize_results.py",
    "class_DeepHit.py",
    "import_data.py",
    "utils_network.py",
]
missing = [p for p in required if not (ROOT / p).exists()]
if missing:
    raise SystemExit("Missing repo files: " + ", ".join(missing))

# ------------------------------------------------------------
# 1) utils_eval_ctd.py = official utils_eval.py + Antolini C^td
# ------------------------------------------------------------
utils = (ROOT / "utils_eval.py").read_text(encoding="utf-8")

ctd_function = r'''
# ============================================================
# Antolini time-dependent concordance index C^td
# Added for the TF control experiment.
# The original repository functions above are unchanged.
# ============================================================
def ctd_index(pred, Time_survival, Death, event=1):
    pred = np.asarray(pred)
    Time_survival = np.asarray(Time_survival).reshape(-1)
    Death = np.asarray(Death).reshape(-1)

    if pred.ndim != 3:
        raise ValueError("pred must have shape [N, num_Event, num_Category]")

    if len(Time_survival) != pred.shape[0] or len(Death) != pred.shape[0]:
        raise ValueError("pred, Time_survival and Death must have the same N")

    event_idx = int(event) - 1
    if event_idx < 0 or event_idx >= pred.shape[1]:
        raise ValueError("event is outside the available event range")

    cif = np.cumsum(pred[:, event_idx, :], axis=1)

    concordant = 0.0
    comparable = 0.0

    for i in range(len(Time_survival)):
        if Death[i] != event:
            continue

        t_i = int(Time_survival[i])
        if t_i < 0 or t_i >= pred.shape[2]:
            raise ValueError(
                "Observed time {} outside prediction horizon [0, {}]".format(
                    t_i, pred.shape[2] - 1
                )
            )

        j_idx = np.where(Time_survival > Time_survival[i])[0]
        if len(j_idx) == 0:
            continue

        risk_i = cif[i, t_i]
        risk_j = cif[j_idx, t_i]

        concordant += np.sum(risk_i > risk_j)
        comparable += len(j_idx)

    if comparable == 0:
        return -1.0

    return float(concordant / comparable)
'''

(ROOT / "utils_eval_ctd.py").write_text(utils + "\n" + ctd_function, encoding="utf-8")

# ------------------------------------------------------------
# 2) get_main_ctd.py = official get_main.py with validation
#    metric changed to global Antolini C^td.
# ------------------------------------------------------------
get_main = (ROOT / "get_main.py").read_text(encoding="utf-8")

old_import = "from utils_eval import c_index, brier_score, weighted_c_index, weighted_brier_score"
new_import = "from utils_eval_ctd import c_index, brier_score, weighted_c_index, weighted_brier_score, ctd_index"
if old_import not in get_main:
    raise SystemExit("Expected utils_eval import not found in get_main.py")
get_main = get_main.replace(old_import, new_import, 1)

old_print = '    print( "EVALUATION TIMES: " + str(eval_time))'
new_print = '    print( "VALIDATION METRIC: Antolini C^td")'
if old_print not in get_main:
    raise SystemExit("Expected evaluation-times print not found in get_main.py")
get_main = get_main.replace(old_print, new_print, 1)

start_marker = "                ### EVALUATION\n"
end_marker = "\n\n                if tmp_valid >  max_valid:"
start = get_main.find(start_marker)
end = get_main.find(end_marker, start)
if start == -1 or end == -1:
    raise SystemExit("Could not locate validation evaluation block in get_main.py")

new_eval = r'''                ### EVALUATION: Antolini C^td
                va_result1 = np.zeros([num_Event])

                for k in range(num_Event):
                    va_result1[k] = ctd_index(
                        pred,
                        va_time[:,0],
                        va_label[:,0],
                        event=k+1
                    )

                    if not np.isfinite(va_result1[k]) or va_result1[k] < 0:
                        raise RuntimeError(
                            "Invalid validation C^td for event {}: {}".format(
                                k+1, va_result1[k]
                            )
                        )

                tmp_valid = np.mean(va_result1)'''

get_main = get_main[:start] + new_eval + get_main[end:]

get_main = get_main.replace(
    "print( 'updated.... average c-index = ' + str('%.4f' %(tmp_valid)))",
    "print( 'updated.... C^td = ' + str('%.4f' %(tmp_valid)))",
    1,
)

(ROOT / "get_main_ctd.py").write_text(get_main, encoding="utf-8")

# ------------------------------------------------------------
# 3) main_RandomSearch_ctd.py = official random search.
#    Only import, output folder and smoke-test budget are changed.
# ------------------------------------------------------------
main = (ROOT / "main_RandomSearch.py").read_text(encoding="utf-8")

if "import get_main" not in main:
    raise SystemExit("Expected get_main import not found in main_RandomSearch.py")
main = main.replace("import get_main", "import get_main_ctd as get_main", 1)

main = main.replace("OUT_ITERATION               = 5", "OUT_ITERATION               = 1", 1)
main = main.replace("RS_ITERATION                = 50", "RS_ITERATION                = 1", 1)
main = main.replace(
    "out_path      = data_mode + '/results/'",
    "out_path      = data_mode + '/results_TF_CTD/'",
    1,
)

(ROOT / "main_RandomSearch_ctd.py").write_text(main, encoding="utf-8")

# ------------------------------------------------------------
# 4) summarize_results_ctd.py = official evaluation script,
#    changed only to METABRIC + C^td + smoke-test budget/path.
# ------------------------------------------------------------
summ = (ROOT / "summarize_results.py").read_text(encoding="utf-8")

old_import_s = "from utils_eval import c_index, brier_score, weighted_c_index, weighted_brier_score"
new_import_s = "from utils_eval_ctd import c_index, brier_score, weighted_c_index, weighted_brier_score, ctd_index"
if old_import_s not in summ:
    raise SystemExit("Expected utils_eval import not found in summarize_results.py")
summ = summ.replace(old_import_s, new_import_s, 1)

summ = summ.replace("OUT_ITERATION               = 5", "OUT_ITERATION               = 1", 1)
summ = summ.replace(
    "data_mode                   = 'SYNTHETIC' #METABRIC, SYNTHETIC",
    "data_mode                   = 'METABRIC' #METABRIC, SYNTHETIC",
    1,
)
summ = summ.replace(
    "in_path = data_mode + '/results/'",
    "in_path = data_mode + '/results_TF_CTD/'",
    1,
)

old_containers = (
    "FINAL1 = np.zeros([num_Event, len(EVAL_TIMES), OUT_ITERATION])\n"
    "FINAL2 = np.zeros([num_Event, len(EVAL_TIMES), OUT_ITERATION])"
)
if old_containers not in summ:
    raise SystemExit("Expected result containers not found in summarize_results.py")
summ = summ.replace(
    old_containers,
    "FINAL_CTD = np.zeros([num_Event, OUT_ITERATION])",
    1,
)

start_marker = "    ### EVALUATION\n"
end_marker = "\n    ### SAVE RESULTS\n"
start = summ.find(start_marker)
end = summ.find(end_marker, start)
if start == -1 or end == -1:
    raise SystemExit("Could not locate test evaluation block in summarize_results.py")

new_test_eval = r'''    ### EVALUATION: Antolini C^td
    result_ctd = np.zeros([num_Event])

    for k in range(num_Event):
        result_ctd[k] = ctd_index(
            pred,
            te_time[:,0],
            te_label[:,0],
            event=k+1
        )

        if not np.isfinite(result_ctd[k]) or result_ctd[k] < 0:
            raise RuntimeError(
                "Invalid test C^td for event {}: {}".format(
                    k+1, result_ctd[k]
                )
            )

    FINAL_CTD[:, out_itr] = result_ctd
'''
summ = summ[:start] + new_test_eval + summ[end:]

save_pos = summ.find("    ### SAVE RESULTS\n")
if save_pos == -1:
    raise SystemExit("Could not locate save-results block in summarize_results.py")

summary_tail = r'''    ### SAVE / PRINT C^td
    df_ctd = pd.DataFrame(
        result_ctd.reshape(-1, 1),
        index=['Event_' + str(k+1) for k in range(num_Event)],
        columns=['Ctd']
    )
    df_ctd.to_csv(
        in_path + '/result_CTD_itr' + str(out_itr) + '.csv'
    )

    print('========================================================')
    print('ITR: ' + str(out_itr+1) + ' DATA MODE: ' + data_mode)
    print('- C^td: ')
    print(df_ctd)
    print('========================================================')


### FINAL MEAN/STD
ctd_mean = np.mean(FINAL_CTD, axis=1)
ctd_std  = np.std(FINAL_CTD, axis=1)

df_ctd_final = pd.DataFrame(
    {
        'mean_Ctd': ctd_mean,
        'std_Ctd': ctd_std
    },
    index=['Event_' + str(k+1) for k in range(num_Event)]
)

df_ctd_final.to_csv(
    in_path + '/result_CTD_FINAL.csv'
)

print('========================================================')
print('- FINAL C^td: ')
print(df_ctd_final)
print('========================================================')
'''

summ = summ[:save_pos] + summary_tail
(ROOT / "summarize_results_ctd.py").write_text(summ, encoding="utf-8")

print("Created:")
print("  utils_eval_ctd.py")
print("  get_main_ctd.py")
print("  main_RandomSearch_ctd.py")
print("  summarize_results_ctd.py")
print("")
print("Original repo Python files were NOT modified.")
