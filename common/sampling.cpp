#include "sampling.h"

#include "common.h"

struct llama_sampling_context * llama_sampling_init(const struct llama_model * model, const struct gpt_sampling_params & params) {
    struct llama_sampling_context * result = new llama_sampling_context();

    result->params = params;

    {
        auto lparams = llama_sampling_default_params();

        lparams.seed              = params.seed;
        lparams.n_prev            = params.n_prev;
        lparams.n_probs           = params.n_probs;
        lparams.min_keep          = params.min_keep;
        lparams.top_k             = params.top_k;
        lparams.top_p             = params.top_p;
        lparams.min_p             = params.min_p;
        lparams.tfs_z             = params.tfs_z;
        lparams.typical_p         = params.typical_p;
        lparams.temp              = params.temp;
        lparams.dynatemp_range    = params.dynatemp_range;
        lparams.dynatemp_exponent = params.dynatemp_exponent;
        lparams.penalty_last_n    = params.penalty_last_n;
        lparams.penalty_repeat    = params.penalty_repeat;
        lparams.penalty_freq      = params.penalty_freq;
        lparams.penalty_present   = params.penalty_present;
        lparams.mirostat          = params.mirostat;
        lparams.mirostat_tau      = params.mirostat_tau;
        lparams.mirostat_eta      = params.mirostat_eta;
        lparams.penalize_nl       = params.penalize_nl;
        lparams.ignore_eos        = params.ignore_eos;

        lparams.n_samplers = params.samplers.size();

        result->smpl = llama_sampling_init(model, lparams);

        llama_sampling_set_grammar   (result->smpl, params.grammar.c_str(), "root");
        llama_sampling_set_logit_bias(result->smpl, params.logit_bias.size(), params.logit_bias.data());
    }

    return result;
}

void llama_sampling_free(struct llama_sampling_context * ctx) {
    llama_sampling_free(ctx->smpl);

    delete ctx;
}

void llama_sampling_reset(llama_sampling_context * ctx) {
    llama_sampling_reset(ctx->smpl);
}

void llama_sampling_cp(llama_sampling_context * src, llama_sampling_context * dst) {
    if (dst->smpl) {
        llama_sampling_free(dst->smpl);
    }

    dst->smpl = llama_sampling_cp(src->smpl);
}

llama_token llama_sampling_last(llama_sampling_context * ctx) {
    return llama_sampling_prev(ctx->smpl, 0);
}

std::string llama_sampling_prev_str(llama_sampling_context * ctx_sampling, llama_context * ctx_main, int n) {
    n = std::min(n, llama_sampling_n_prev(ctx_sampling->smpl));

    if (n <= 0) {
        return "";
    }

    std::string result;
    result.reserve(8*n); // 8 is the average length of a token [citation needed], TODO: compute this from the vocab

    for (int i = n - 1; i >= 0; i--) {
        const llama_token id = llama_sampling_prev(ctx_sampling->smpl, i);

        GGML_ASSERT(id != LLAMA_TOKEN_NULL && "null token in the sampling history - should not happen");

        result += llama_token_to_piece(ctx_main, id);
    }

    return result;
}

std::string llama_sampling_print(const gpt_sampling_params & params) {
    char result[1024];

    snprintf(result, sizeof(result),
            "\trepeat_last_n = %d, repeat_penalty = %.3f, frequency_penalty = %.3f, presence_penalty = %.3f\n"
            "\ttop_k = %d, tfs_z = %.3f, top_p = %.3f, min_p = %.3f, typical_p = %.3f, temp = %.3f\n"
            "\tmirostat = %d, mirostat_lr = %.3f, mirostat_ent = %.3f",
            params.penalty_last_n, params.penalty_repeat, params.penalty_freq, params.penalty_present,
            params.top_k, params.tfs_z, params.top_p, params.min_p, params.typical_p, params.temp,
            params.mirostat, params.mirostat_eta, params.mirostat_tau);

    return std::string(result);
}

std::string llama_sampling_order_print(const gpt_sampling_params & params) {
    std::string result = "CFG -> Penalties ";
    if (params.mirostat == 0) {
        for (auto sampler_type : params.samplers) {
            const auto sampler_type_name = llama_sampling_type_to_str(sampler_type);
            if (!sampler_type_name.empty()) {
                result += "-> " + sampler_type_name + " ";
            }
        }
    } else {
        result += "-> mirostat ";
    }

    return result;
}

char llama_sampling_type_to_chr(llama_sampler_type sampler_type) {
    switch (sampler_type) {
        case LLAMA_SAMPLER_TYPE_TOP_K:       return 'k';
        case LLAMA_SAMPLER_TYPE_TFS_Z:       return 'f';
        case LLAMA_SAMPLER_TYPE_TYPICAL_P:   return 'y';
        case LLAMA_SAMPLER_TYPE_TOP_P:       return 'p';
        case LLAMA_SAMPLER_TYPE_MIN_P:       return 'm';
        case LLAMA_SAMPLER_TYPE_TEMPERATURE: return 't';
        default : return '?';
    }
}

std::string llama_sampling_type_to_str(llama_sampler_type sampler_type) {
    switch (sampler_type) {
        case LLAMA_SAMPLER_TYPE_TOP_K:       return "top_k";
        case LLAMA_SAMPLER_TYPE_TFS_Z:       return "tfs_z";
        case LLAMA_SAMPLER_TYPE_TYPICAL_P:   return "typical_p";
        case LLAMA_SAMPLER_TYPE_TOP_P:       return "top_p";
        case LLAMA_SAMPLER_TYPE_MIN_P:       return "min_p";
        case LLAMA_SAMPLER_TYPE_TEMPERATURE: return "temperature";
        default : return "";
    }
}

std::vector<llama_sampler_type> llama_sampling_types_from_names(const std::vector<std::string> & names, bool allow_alt_names) {
    std::unordered_map<std::string, llama_sampler_type> sampler_canonical_name_map {
        { "top_k",       LLAMA_SAMPLER_TYPE_TOP_K },
        { "top_p",       LLAMA_SAMPLER_TYPE_TOP_P },
        { "typical_p",   LLAMA_SAMPLER_TYPE_TYPICAL_P },
        { "min_p",       LLAMA_SAMPLER_TYPE_MIN_P },
        { "tfs_z",       LLAMA_SAMPLER_TYPE_TFS_Z },
        { "temperature", LLAMA_SAMPLER_TYPE_TEMPERATURE },
    };

    // since samplers names are written multiple ways
    // make it ready for both system names and input names
    std::unordered_map<std::string, llama_sampler_type> sampler_alt_name_map {
        { "top-k",       LLAMA_SAMPLER_TYPE_TOP_K },
        { "top-p",       LLAMA_SAMPLER_TYPE_TOP_P },
        { "nucleus",     LLAMA_SAMPLER_TYPE_TOP_P },
        { "typical-p",   LLAMA_SAMPLER_TYPE_TYPICAL_P },
        { "typical",     LLAMA_SAMPLER_TYPE_TYPICAL_P },
        { "min-p",       LLAMA_SAMPLER_TYPE_MIN_P },
        { "tfs-z",       LLAMA_SAMPLER_TYPE_TFS_Z },
        { "tfs",         LLAMA_SAMPLER_TYPE_TFS_Z },
        { "temp",        LLAMA_SAMPLER_TYPE_TEMPERATURE },
    };

    std::vector<llama_sampler_type> sampler_types;
    sampler_types.reserve(names.size());

    for (const auto & name : names) {
        auto sampler_item = sampler_canonical_name_map.find(name);
        if (sampler_item != sampler_canonical_name_map.end()) {
            sampler_types.push_back(sampler_item->second);
        } else {
            if (allow_alt_names) {
                sampler_item = sampler_alt_name_map.find(name);
                if (sampler_item != sampler_alt_name_map.end()) {
                    sampler_types.push_back(sampler_item->second);
                }
            }
        }
    }

    return sampler_types;
}

std::vector<llama_sampler_type> llama_sampling_types_from_chars(const std::string & names_string) {
    std::unordered_map<char, llama_sampler_type> sampler_name_map {
        { llama_sampling_type_to_chr(LLAMA_SAMPLER_TYPE_TOP_K),       LLAMA_SAMPLER_TYPE_TOP_K },
        { llama_sampling_type_to_chr(LLAMA_SAMPLER_TYPE_TFS_Z),       LLAMA_SAMPLER_TYPE_TFS_Z },
        { llama_sampling_type_to_chr(LLAMA_SAMPLER_TYPE_TYPICAL_P),   LLAMA_SAMPLER_TYPE_TYPICAL_P },
        { llama_sampling_type_to_chr(LLAMA_SAMPLER_TYPE_TOP_P),       LLAMA_SAMPLER_TYPE_TOP_P },
        { llama_sampling_type_to_chr(LLAMA_SAMPLER_TYPE_MIN_P),       LLAMA_SAMPLER_TYPE_MIN_P },
        { llama_sampling_type_to_chr(LLAMA_SAMPLER_TYPE_TEMPERATURE), LLAMA_SAMPLER_TYPE_TEMPERATURE }
    };

    std::vector<llama_sampler_type> sampler_types;
    sampler_types.reserve(names_string.size());
    for (const auto & c : names_string) {
        const auto sampler_item = sampler_name_map.find(c);
        if (sampler_item != sampler_name_map.end()) {
            sampler_types.push_back(sampler_item->second);
        }
    }
    return sampler_types;
}

// no reasons to expose this function in header
static void sampler_queue(
          struct llama_sampling_context * ctx_sampling,
          struct llama_token_data_array * cur_p) {
    llama_sampling * smpl = ctx_sampling->smpl;

    const gpt_sampling_params & params = ctx_sampling->params;

    const std::vector<llama_sampler_type> & samplers = params.samplers;

    for (const auto & sampler : samplers) {
        switch (sampler) {
            case LLAMA_SAMPLER_TYPE_TOP_K:       llama_sampling_top_k    (smpl, cur_p); break;
            case LLAMA_SAMPLER_TYPE_TFS_Z:       llama_sampling_tail_free(smpl, cur_p); break;
            case LLAMA_SAMPLER_TYPE_TYPICAL_P:   llama_sampling_typical  (smpl, cur_p); break;
            case LLAMA_SAMPLER_TYPE_TOP_P:       llama_sampling_top_p    (smpl, cur_p); break;
            case LLAMA_SAMPLER_TYPE_MIN_P:       llama_sampling_min_p    (smpl, cur_p); break;
            case LLAMA_SAMPLER_TYPE_TEMPERATURE: llama_sampling_temp     (smpl, cur_p); break;
            default : break;
        }
    }
}

void llama_sampling_prepare(
        struct llama_sampling_context * ctx_sampling,
        struct llama_context * ctx_main,
        int idx) {
    llama_sampling_set_logits(ctx_sampling->smpl, llama_get_logits_ith(ctx_main, idx));
}

static llama_token llama_sampling_sample(
        struct llama_sampling_context * ctx_sampling,
        struct llama_token_data_array * cur_p) {
    llama_sampling * smpl = ctx_sampling->smpl;

    const gpt_sampling_params & params = ctx_sampling->params;

    const float temp     = params.temp;
    const int   mirostat = params.mirostat;

    llama_token id = 0;

    if (temp < 0.0f || (temp == 0.0f && params.n_probs > 0)) {
        // greedy sampling, with probs
        llama_sampling_softmax(smpl, cur_p);
        id = cur_p->data[0].id;
    } else if (temp == 0.0f) {
        // greedy sampling, no probs
        id = llama_sampling_sample_greedy(smpl, cur_p);
    } else {
        if (mirostat != 0) {
            llama_sampling_temp(smpl, cur_p);
            id = llama_sampling_sample_mirostat(smpl, cur_p);
        } else {
            sampler_queue(ctx_sampling, cur_p);

            id = llama_sampling_sample_dist(smpl, cur_p);

            //{
            //    const int n_top = 10;
            //    LOG("top %d candidates:\n", n_top);

            //    for (int i = 0; i < n_top; i++) {
            //        const llama_token id = cur_p.data[i].id;
            //        (void)id; // To avoid a warning that id is unused when logging is disabled.
            //        LOG(" - %5d: '%12s' (%.3f)\n", id, llama_token_to_piece(smpl, id).c_str(), cur_p.data[i].p);
            //    }
            //}

            //LOG("sampled token: %5d: '%s'\n", id, llama_token_to_piece(smpl, id).c_str());
        }
    }

    return id;
}

llama_token llama_sampling_sample(
        struct llama_sampling_context * ctx_sampling,
        struct llama_context * ctx_main,
        int idx) {
    llama_sampling_prepare(ctx_sampling, ctx_main, idx);

    auto * cur_p = llama_sampling_get_candidates(ctx_sampling->smpl);

    llama_sampling_grammar(ctx_sampling->smpl, cur_p);

    return llama_sampling_sample(ctx_sampling, cur_p);
}

void llama_sampling_accept(
        struct llama_sampling_context * ctx_sampling,
        llama_token id,
        bool apply_grammar) {
    llama_sampling_accept(ctx_sampling->smpl, id, apply_grammar);
}
