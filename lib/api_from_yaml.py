import json

import yaml


def save_descriptions(key, param, descrs):
    if not key in descrs.keys():
        descrs[key] = []

    s = set(descrs[key])
    s.add(param)
    descrs[key] = list(s)


def generate_json_from_schema(schema, descrs, descr_res):
    """
    Recursively generate a JSON template from OpenAPI/YAML schema definition.
    """

    # Base primitive types → placeholder values
    placeholder = {
        "string": "text",
        "integer": 1,
        "number": 1,
        "boolean": True
    }

    # if not isinstance(schema, dict):
    #     return None

    schema_type = schema.get("type")
    print(schema_type)

    # ---- Case 1: Object ----
    if schema_type == "object":
        props = schema.get("properties", {})
        result = {}
        for key, value in props.items():
            result[key] = generate_json_from_schema(value, descrs, descr_res)

            if "description" in value.keys():
                descr_res[key] = descrs["response"][key]
        return result

    # ---- Case 2: Array ----
    if schema_type == "array":
        items = schema.get("items", {})
        return [generate_json_from_schema(items, descrs, descr_res)]

    # ---- Case 3: Enum → return first value ----
    if "enum" in schema:
        return schema["enum"][0] if schema["enum"] else None

    # ---- Case 5: Format overrides (binary/byte serialized fields) ----
    if schema.get("format") == "date-time":
        return "2025-11-16T12:37:10.832Z"

    if schema.get("format") == "byte":
        return "Ynl0ZXM="

    # ---- Case 4: Primitive types ----
    if schema_type in placeholder:
        return placeholder[schema_type]

    # ---- Case 6: Unknown → return None ----
    return None


with open('../data/juno.yml') as f:
    # use safe_load instead load
    dataMap = yaml.safe_load(f)
    count = 0
    res = {}
    with open("/root/neutron/IAMY/data/descrs_short.txt", 'r') as f:
        descrs_short = json.load(f)

    for i in dataMap['paths']:
        for key, val in dataMap['paths'][i].items():
            res[i] = {"parameters": [], "responses": {}, "descriptions": {}}
            for ent in val["responses"]["200"]["content"]["application/json"]:
                res[i]["responses"] = generate_json_from_schema(
                    val["responses"]["200"]["content"]["application/json"]["schema"], descrs_short, res[i]["descriptions"])

            if key == "get":
                if "parameters" in val.keys():
                    for p in val["parameters"]:
                        obj = {"name": p["name"], "required": p["required"], "type": p["schema"]["type"]}
                        if "description" in p.keys():
                            obj["description"] = descrs_short["request"][p["name"]]
                        res[i]["parameters"].append(obj)

            elif key == "post":
                res[i]["parameters"] = generate_json_from_schema(
                    val["requestBody"]["content"]["application/json"]["schema"], descrs_short, res[i]["descriptions"])
                # print(val['summary'])
                # else:
                #     print(key)
                #     die

    with open('../data/res.json', 'w') as f:
        json.dump(res, f, indent=4)
