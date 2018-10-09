(TeX-add-style-hook
 "main"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("acmart" "sigconf" "anonymous")))
   (TeX-run-style-hooks
    "latex2e"
    "macros"
    "acmart"
    "acmart10")
   (LaTeX-add-labels
    "sec_bayesInfer"
    "sec_setup"
    "subsec_emgs"
    "equ_utility"
    "subsec_emls"
    "sec_smoo"
    "def_smoo"
    "lem_hexpmech_privacy"
    "subsec_accuracy_lap"
    "subsec_accuracy_global"
    "subsec_accuracy_smoo"
    "subsec_accuracy_tradeoff"
    "tab_8"
    "tab_40"
    "fig_sensitivity"
    "fig_efficiency"
    "sec_experiment"
    "subsec_effi"
    "subsec_vs_variables"
    "subsubsec_vs_datasize"
    "subsubsec_vs_datasize1a"
    "subsubsec_vs_datasize1b"
    "fig_vs_datasize"
    "subsubsec_vs_datasize1bdir"
    "fig_vs_datasize_dir"
    "subsubsec_vs_prior"
    "fig_vs_prior"
    "subsec_experiment_privacy"
    "fig_privacy")
   (LaTeX-add-bibliographies
    "bayesian.bib"))
 :latex)

