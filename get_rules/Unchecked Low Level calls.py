def identify_unchecked_low_level_calls(json_ast):
    
    main_objective_functions = []

    def has_verification_after_call(node):
        # Check if there is a verification statement after a low-level call
        if isinstance(node, list):
            for i, item in enumerate(node):
                if isinstance(item, dict) and item.get("name") in ["call", "callcode", "delegatecall", "send"]:
                    # Check next statements for verification
                    for j in range(i+1, len(node)):
                        next_item = node[j]
                        if isinstance(next_item, dict) and next_item.get("type") in ["IfStatement", "RequireStatement", "AssertStatement"]:
                            return True
                    return False
                if has_verification_after_call(item):
                    return True
        elif isinstance(node, dict):
            for value in node.values():
                if has_verification_after_call(value):
                    return True
        return False

    def scan_function(node):
        # Scan a function node for unchecked low level calls
        if isinstance(node, dict) and node.get("type") == "FunctionDefinition":
            function_name = node.get("name")
            if not has_verification_after_call(node):
                main_objective_functions.append(function_name)
            for value in node.values():
                if isinstance(value, (dict, list)):
                    scan_function(value)
        elif isinstance(node, list):
            for item in node:
                scan_function(item)

    scan_function(json_ast)
    return main_objective_functions

