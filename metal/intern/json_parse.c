// json_parse
using json_value = std::variant<int, float, bool, std::string>;
static std::map<std::string, json_value>
json_parse_flat_object(const std::string& txt)
{
    std::map<std::string, json_value> out;
    std::string key, val;
    bool in_key = false, in_val = false;
    bool in_string = false;

    for (size_t i = 0; i < txt.size(); ++i) {
        char c = txt[i];

        if (c == '"') {
            in_string = !in_string;
            continue;
        }

        if (in_string) {
            if (!in_val && !in_key)
                key.push_back(c);
            else
                val.push_back(c);
            continue;
        }

        if (c == ':') {
            in_key = false;
            in_val = true;
            continue;
        }

        if (c == ',') {
            // finalize value
            std::stringstream ss(val);
            if (val == "true" || val == "false") {
                out[key] = (val == "true");
            } else if (val.find('.') != std::string::npos) {
                float f;
                ss >> f;
                out[key] = f;
            } else {
                int n;
                ss >> n;
                out[key] = n;
            }
            key.clear();
            val.clear();
            in_val = false;
            continue;
        }

        if (c == '{' || c == '}' || c == ' ' || c == '\n') continue;

        if (!in_key && !in_val) {
            in_key = true;
            continue;
        }

        if (in_key) key.push_back(c);
        else if (in_val) val.push_back(c);
    }

    // finalize last key/value
    if (!key.empty() && !val.empty()) {
        std::stringstream ss(val);
        if (val == "true" || val == "false")
            out[key] = (val == "true");
        else if (val.find('.') != std::string::npos) {
            float f; ss >> f; out[key] = f;
        } else {
            int n; ss >> n; out[key] = n;
        }
    }

    return out;
}

// Simple Dtype → string helper for error messages
const char* dtype_to_string(mx::Dtype dt) {
    switch (dt) {
        case mx::float16: return "float16";
        case mx::float32: return "float32";
        case mx::float64: return "float64";
        case mx::int32:   return "int32";
        case mx::int64:   return "int64";
        case mx::uint8:   return "uint8";
        // Add more if you use them; default keeps us safe.
        default:          return "unknown";
    }
}

// Read whole file into string
std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.good()) return {};
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Build a full tensor key like "model.layers.<layer>.<suffix>"
std::string layer_key(int layer, const char* suffix) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "model.layers.%d.%s", layer, suffix);
    return std::string(buf);
}

