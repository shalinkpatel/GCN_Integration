using DrWatson, PyCall
@quickactivate "GCN_HM_GRN-Integration"

pushfirst!(PyVector(pyimport("sys")."path"), "");
run_sim = pyimport("inner_batch.py").run_sim

allparams = Dict(
    :layer => ["arma", "sage", "tag"], 
    :batch => [20, 30],
    :cl => ["E116"]
)

dicts = dict_list(allparams)

function makesim(d::Dict)
    @unpack layer, batch, cl = d
    best_auc, losses_test, losses_train, auc_l = run_sim(cl, batch, layer)
    fulld = copy(d)
    fulld[:auc] = best_auc
    fulld[:losses_test] = losses_test
    fulld[:losses_train] = losses_train
    fulld[:auc_list] = auc_l
    return fulld
end

for (i, d) in enumerate(dicts)
    f = makesim(d)
    print("Finished $(i)")
    @tagsave(datadir("inner_batch", savename(d, "bson")), f; safe = true)
end

