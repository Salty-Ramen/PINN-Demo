#importing existing dataset and generating samples
using CSV, DataFrames
using Pkg
using XLSX
project_root_dir = dirname(Pkg.project().path)
viral_load_path = joinpath(project_root_dir, "Data-Master", "Viral Load.xlsx")
states_path = joinpath(project_root_dir, "Data-Master", "States.xlsx")

df_viral_load = XLSX.openxlsx(viral_load_path) do xf
    sh   = xf["Robinson et al., IL17..."]     # sheet by name
    df  = DataFrame(XLSX.gettable(sh))
end

df_states = XLSX.openxlsx(states_path) do xf
    sh   = xf["Robinson et al., IL17..."]     # sheet by name
    df  = DataFrame(XLSX.gettable(sh))
end

# generate samples
using Random
using Statistics

rng = Random.MersenneTwister(42)
# This is vcat with a array comprehension syntax. 
# iterating over each row (r being the row)
# fill here makes a vector of length r.n for correct sizing
# the splat operator inside vcat is used to convert an array of DataFrames into a tall one
df_viral_samples = vcat([
        DataFrame(
            Symbol("Time (Days)") => fill(r.var"Time (Days)", r.n),
            :Virus => r.var"log Virus (TCID50/mL)" .+ r.logVirusstd .* randn(r.n),
            :Sex => fill(r.Sex, r.n),
            :SampleID => 1:r.n,
        )
        for r in eachrow(df_viral_load)
]...);

df_viral_samples = stack(
    df_viral_samples, :Virus;
    variable_name = :State,
    value_name = :StateSamples
)

id_cols = [
    Symbol("Time (Days)"),
    :n,
    :Sex,
    Symbol("Viral Strain"),
    Symbol("Mice Species")
]

stat_cols = names(df_states, Not(id_cols))  # colnames not belonging to id_cols

# stack() is just like pivot_longer() from tidyverse in R
long = stack(
    df_states, stat_cols;
    variable_name= :variables,
    value_name= :values
)

# :variables => ByRow(f) => AsTable
# reads: apply function f to each row's :variable entry
# AsTable expands the named tuple as a column entry

transform!( #in place mutation 
    long,
    :variables => ByRow(
        s -> begin
            col = String(s)
            m = match(r"^(.*?)(stdev|Norm)?$", col)  # pattern match colname
            base = m.captures[1] # splitting colname (m) into base and raw
            raw = m.captures[2]
            stat = isnothing(raw) ? "mean" : lowercase(raw) # a simple if else for stat
            (state = base, statistic = stat) # returns this tuple for each row
        end
    ) => AsTable
)

# Gymnastics
long.values = replace(long.values,
    r"^\s*$" => missing,   # empty / whitespace-only
    "NA"     => missing,
    "N/A"    => missing,
    "NaN"    => missing
)

# force values to be numeric
long.values = passmissing(x -> parse(Float64, string(x))).(long.values)

#drop NA
dropmissing!(long, :values)

# unstack is like pivot_wider()
df_long = unstack(
    long, vcat(id_cols, [:state]), :statistic, :values
)

subset!(
    df_long,
    :state => ByRow(
        s -> lowercase(s) in lowercase.(["CCL2", "IL-6"])
    )
)

df_state_samples = vcat([
        DataFrame(
            Symbol("Time (Days)") => fill(r.var"Time (Days)", r.n),
            :State => fill(r.state, r.n),
            :StateSamples => clamp.(r.mean .+ r.stdev .* randn(rng, r.n), 0, Inf),
            :Sex => fill(r.Sex, r.n),
            :SampleID => 1:r.n,
        )
        for r in eachrow(df_long)
]...)

df_samples = vcat([df_viral_samples, df_state_samples]...);


## Generate NN ready Dataset
# Test only for females
df_samples_female = subset(
    df_samples,
    :Sex => ByRow(
        in(["F"])
    )
) 

df_samples_female = unstack(
    df_samples_female,
    [Symbol("Time (Days)"), :SampleID],
    :State,
    :StateSamples
) 

CSV.write(joinpath(project_root_dir, "Results", "Robinson_train.csv"), df_samples)
CSV.write(joinpath(project_root_dir, "Results", "Robinson_train_female.csv"), df_samples_female)
