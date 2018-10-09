(TeX-add-style-hook
 "main-sketch"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("acmart" "sigconf")))
   (TeX-run-style-hooks
    "latex2e"
    "macros"
    "acmart"
    "acmart10"
    "accents")
   (LaTeX-add-labels
    "sec_intro"
    "sec_background"
    "sec:base"
    "sec_smoo"
    "eq:smooth"
    "thm:privacy"
    "sec_experiment"
    "subsubsec_vs_datasize"
    "subsubsec_vs_datasize1a"
    "subsubsec_vs_datasize1b"
    "fig_vs_datasize"
    "subsubsec_vs_datasize1bdir"
    "fig_vs_datasize_dir"
    "subsubsec_vs_prior"
    "fig_vs_prior")
   (LaTeX-add-bibliographies
    "bayesian.bib"))
 :latex)

