using IteratedProcessSimulations
using Documenter
using VegaLite, UUIDs

# Snippet source: https://github.com/queryverse/VegaLite.jl/blob/master/docs/make.jl
function Base.show(io::IO, m::MIME"text/html", v::VegaLite.VLSpec)
    divid = string("vl", replace(string(uuid4()), "-" => ""))
    print(io, "<div id='$divid' style=\"width:100%;height:100%;\"></div>")
    print(io, "<script type='text/javascript'>requirejs.config({paths:{'vg-embed': 'https://cdn.jsdelivr.net/npm/vega-embed@6?noext','vega-lib': 'https://cdn.jsdelivr.net/npm/vega-lib?noext','vega-lite': 'https://cdn.jsdelivr.net/npm/vega-lite@4?noext','vega': 'https://cdn.jsdelivr.net/npm/vega@5?noext'}}); require(['vg-embed'],function(vegaEmbed){vegaEmbed('#$divid',")    
    VegaLite.our_json_print(io, v)
    print(io, ",{mode:'vega-lite'}).catch(console.warn);})</script>")
end
# End snippet

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
