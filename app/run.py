from flaskr import create_app


app = create_app()

if __name__ == "__main__":
    host = app.config.get("FLASK_RUN_HOST", "127.0.0.1")
    port = app.config.get("FLASK_RUN_PORT", 5000)
    debug = app.config.get("FLASK_DEBUG", False)
    print(f"Running on {host}:{port} with debug={debug}")
    
    app.run(host=host, port=port, debug=debug)
