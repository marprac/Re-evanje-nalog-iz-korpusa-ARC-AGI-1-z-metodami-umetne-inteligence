# Resevanje-nalog-iz-korpusa-ARC-AGI-1-z-metodami-umetne-inteligence
Glavni program je zasnovan za izvajanje in vrednotenje sistema na nalogah iz nabora ARC. Program uporablja knjižnico \texttt{DSPy} za vodenje velikih jezikovnih modelov (LLM) in uvede mehanizme za ustvarjanje, popravljanje ter dodajanje navodil.

\paragraph{Vhodi programa:}
\begin{itemize}
    \item \textbf{Naloge v formatu JSON} -- vsaka naloga vsebuje učne pare
    (vhodna in izhodna matrika) ter testne primere. Pot do teh datotek moramo podati v kodi.
    \item \textbf{Model LLM} -- v konfiguraciji se določi uporabljeni model
    (npr. \texttt{gpt-5-mini}), temperatura in število tokenov.
\end{itemize}

\paragraph{Izhodi programa:}
\begin{itemize}
    \item \textbf{Navodila v naravnem jeziku} -- začetni seznam predpogojev in
    akcij, ki opisujejo transformacijo.
    \item \textbf{Popravljena navodila} -- iterativno izboljšan seznam
    navodil, ki jih program ustvari ob neuspehu začetne rešitve.
    \item \textbf{Transformirane matrike} -- rezultat izvajanja navodil na
    vhodnih in testnih primerih.
    \item \textbf{Poročila in log datoteke} -- vsaka naloga ima svoj izpis
    (\texttt{.txt}), kjer se beležijo začetna navodila, spremembe med
    popravljanjem, razlike med pričakovano in dobljeno matriko ter končni
    rezultat. V posebno datoteko se beleži končen povzetek uspešnosti (število rešenih nalog).
\end{itemize}

V kodo moramo dodati poti do primerov nalog, direktorija za sharnjevanje informacij o posamezni nalogi in datoteke za shranjevanje končnih informacij. Originalna koda se nahaja na javnem GitHub repozitoriju.
