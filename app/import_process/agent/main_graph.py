if __name__ == '__main__':
    from app.import_process.agent.kb_import_workflow import KBImportWorkflow
    from app.import_process.agent.state import create_default_state
    kb_import_app = KBImportWorkflow().compile()
    init_state = create_default_state(
        task_id="task_001",
        local_file_path="d:/abc.pdf"
    )

    for chunk in kb_import_app.stream(init_state):
        print(type(chunk))
        print(chunk.keys())
        print(chunk.items())
