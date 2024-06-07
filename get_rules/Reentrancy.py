def identify_reentrancy(json_ast):

    main_objective_functions = []

    def contains_balance_changing_after_transfer(statements):
        # check if balance-changing occurs after a transfer statement
        transfer_found = False
        for statement in statements:
            if isinstance(statement, dict):
                # Check if the statement is a transfer statement
                if statement.get("type") == "TransferStatement" or ('call.value' in statement.get("name", "") and 'msg.sender' in statement.get("name", "")):
                    # Skip zero-value transfers
                    if 'value(0)' in statement.get("name", ""):
                        continue
                    transfer_found = True
                elif transfer_found and 'balance' in statement.get("name", ""):
                    # Balance changing statement found after transfer
                    return True
        return False

    def scan_function(node):
        # Scan a function node for transfer and balance-changing statements
        if isinstance(node, dict):
            if node.get("type") == "FunctionDefinition":
                body_statements = node.get("body", {}).get("statements", [])
                if contains_balance_changing_after_transfer(body_statements):
                    main_objective_functions.append(node.get("id", node.get("name")))
            for value in node.values():
                if isinstance(value, (list, dict)):
                    scan_function(value)
        elif isinstance(node, list):
            for item in node:
                scan_function(item)

    scan_function(json_ast)
    return main_objective_functions

