using DrWatson, PyCall
@quickactivate "GCN_HM_GRN-Integration"

pushfirst!(PyVector(pyimport("sys")."path"), "");
run_sim = pyimport("port").run_sim

allparams = Dict(
    :k => [2, 3], 
    :epochs => [500, 750],
    :cl => ["E116"]
)

dicts = dict_list(allparams)

function makesim(d::Dict)
    @unpack k, epochs, cl = d
    best_auc = run_sim(cl, epochs, k)
    fulld = copy(d)
    fulld[:auc] = best_auc
    return fulld
end

for (i, d) in enumerate(dicts)
    f = makesim(d)
    print("Finished $(i)")
    @tagsave(datadir("port_no_sep", savename(d, "bson")), f)
end