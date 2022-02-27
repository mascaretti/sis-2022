.PHONY: all compile clean distclean

all:
	latexmk --pdf author.tex & xdg-open author.pdf &

compile:
	latexmk --pdf author.tex

clean:
	latexmk -c author.tex

distclean:
	latexmk -C author.tex
