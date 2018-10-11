(TeX-add-style-hook
 "main"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("beamer" "final" "hyperref={pdfpagelabels=false}")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("beamerposter" "orientation=landscape" "size=custom" "width=101.6" "height=76.2" "scale=1.4") ("babel" "english")))
   (add-to-list 'LaTeX-verbatim-environments-local "semiverbatim")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "macros"
    "beamer"
    "beamer10"
    "beamerposter"
    "proof"
    "babel"
    "amsmath"
    "amsthm"
    "amssymb"
    "latexsym"
    "booktabs"
    "stmaryrd")
   (TeX-add-symbols
    '("stmod" 1)
    '("interp" 2)
    "leftfoot"
    "rightfoot")
   (LaTeX-add-labels
    "fig_sensitivity"
    "subfig_concrete_prob_2d"
    "subfig_concrete_prob_3d"
    "subfig_prior"
    "subfig_sampling_2d"
    "subfig_sampling_3d"
    "subfig_sampling_4d"
    "fig_sampling")
   (LaTeX-add-bibliographies
    "bayesian"))
 :latex)

