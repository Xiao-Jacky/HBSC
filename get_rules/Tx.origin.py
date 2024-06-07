def identify_tx_origin(json_ast):

    main_objective_functions = []
    modifiers_with_tx_origin = []

    def scan_for_tx_origin(node, within_modifier=False):
        if isinstance(node, dict):
            if node.get("type") == "ModifierDefinition":
                has_tx_origin = scan_for_tx_origin(node.get("body"), within_modifier=True)
                if has_tx_origin:
                    modifiers_with_tx_origin.append(node.get("name"))
            elif node.get("type") == "Identifier" and node.get("name") == "tx.origin":
                return True
            for key, value in node.items():
                if isinstance(value, (dict, list)) and scan_for_tx_origin(value, within_modifier):
                    return True
        elif isinstance(node, list):
            for item in node:
                if scan_for_tx_origin(item, within_modifier):
                    return True
        return False

    def scan_for_usage_in_modifiers(node):
        # Check if function uses a modifier that contains "tx.origin"
        if isinstance(node, dict):
            if node.get("type") == "FunctionDefinition":
                # Check for specific condition "msg.sender == tx.origin"
                if not ('msg.sender == tx.origin' in str(node.get("body"))):
                    function_modifiers = [mod.get("name") for mod in node.get("modifiers", []) if mod.get("type") == "ModifierInvocation"]
                    if any(mod in modifiers_with_tx_origin for mod in function_modifiers):
                        main_objective_functions.append(node.get("name"))
            for value in node.values():
                if isinstance(value, (dict, list)):
                    scan_for_usage_in_modifiers(value)
        elif isinstance(node, list):
            for item in node:
                scan_for_usage_in_modifiers(item)

    scan_for_tx_origin(json_ast)
    scan_for_usage_in_modifiers(json_ast)
    return main_objective_functions

