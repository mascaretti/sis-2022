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
    "sec:2"
    "eq:01"
    "subsec:2"
    "fig:1"
    "fig:2"
    "tab:1"
    "sec:3")
   (LaTeX-add-index-entries
    "cross-references"
    "citations"
    "permission to print"
    "paragraph"))
 :latex)

