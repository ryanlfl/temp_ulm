from ulm_frf_app_datalake_v4 import app

app = app
server = app.server
if(__name__=='__main__'):
    # app.run_server(debug=True, port=8080, host='0.0.0.0')
    server.run(debug=True, port=8002)
