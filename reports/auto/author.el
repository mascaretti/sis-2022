(TeX-add-style-hook
 "author"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("svmult" "graybox")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("footmisc" "bottom")))
   (TeX-run-style-hooks
    "latex2e"
    "referenc"
    "svmult"
    "svmult10"
    "mathptmx"
    "helvet"
    "courier"
    "type1cm"
    "makeidx"
    "graphicx"
    "multicol"
    "footmisc")
   (LaTeX-add-labels
    "sec:1"
    "eq:1"
    "eq:2"
    "eq:3"
    "eq:6"))
 :latex)

