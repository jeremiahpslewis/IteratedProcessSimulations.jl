using IteratedProcessSimulations
using Documenter

DocMeta.setdocmeta!(IteratedProcessSimulations, :DocTestSetup, :(using IteratedProcessSimulations); recursive=true)

makedocs(;
    modules=[IteratedProcessSimulations],
    authors="Jeremiah Lewis <4462211+jlewis91@users.noreply.github.com> and contributors",
    repo="https://github.com/jeremiahpslewis/IteratedProcessSimulations.jl/blob/{commit}{path}#{line}",
    sitename="IteratedProcessSimulations.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://jeremiahpslewis.github.io/IteratedProcessSimulations.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Recommender Vignette" => "Vignette-Recommender.md",
    ],
)

deploydocs(;
    repo="github.com/jeremiahpslewis/IteratedProcessSimulations.jl",
    devbranch="main",
)
