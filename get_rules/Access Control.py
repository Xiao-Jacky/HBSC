def identify_access_control(json_ast):

    main_objective_functions = []

    def has_proper_access_control(node):
        # Check for specific modifiers or conditional statements that provide access control
        if isinstance(node, list):
            for statement in node:
                if isinstance(statement, dict) and statement.get("type") in ["IfStatement", "RequireStatement", "AssertStatement"]:
                    return True
        elif isinstance(node, dict):
            for key, value in node.items():
                if has_proper_access_control(value):
                    return True
        return False

    def scan_function(node):

        if isinstance(node, dict) and node.get("type") == "FunctionDefinition":
            visibility = node.get("visibility", "public")  # Assume public if not specified
            if visibility in ["public", "external"]:
                body_statements = node.get("body", {}).get("statements", [])
                # Check if the function contains transfer or sensitive operations
                if any("transfer" in statement.get("name", "") or "sensitive" in statement.get("name", "") for statement in body_statements):
                    if not has_proper_access_control(body_statements):
                        main_objective_functions.append(node.get("id", node.get("name")))
        for value in node.values():
            if isinstance(value, (list, dict)):
                scan_function(value)

    scan_function(json_ast)
    return main_objective_functions

