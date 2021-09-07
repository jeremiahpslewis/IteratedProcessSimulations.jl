using MLBizOps
using Documenter

DocMeta.setdocmeta!(MLBizOps, :DocTestSetup, :(using MLBizOps); recursive=true)

makedocs(;
    modules=[MLBizOps],
    authors="Jeremiah Lewis <4462211+jlewis91@users.noreply.github.com> and contributors",
    repo="https://github.com/jeremiahpslewis/MLBizOps.jl/blob/{commit}{path}#{line}",
    sitename="MLBizOps.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://jeremiahpslewis.github.io/MLBizOps.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jeremiahpslewis/MLBizOps.jl",
    devbranch="main",
)
