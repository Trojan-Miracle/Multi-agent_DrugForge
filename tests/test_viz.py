import asyncio
import http.client
import json
import threading
import pytest
import DrugForge as app

@pytest.fixture
def viewer(monkeypatch):
    monkeypatch.setattr(app, '_run', None)
    monkeypatch.setattr(app, '_pending_approval', None)
    server = app.HTTPServer(('127.0.0.1', 0), app.VizHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_port
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)

def request(port, method, path, value=None, headers=None):
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=3)
    try:
        connection.request(method, path, json.dumps(value) if value is not None else None, headers or {})
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()

def test_unknown_routes_and_stale_confirmation(viewer):
    assert request(viewer, 'GET', '/missing')[0] == 404
    assert request(viewer, 'POST', '/missing')[0] == 404
    assert request(viewer, 'POST', '/confirm', {'token': 'old'})[0] == 409
    assert request(viewer, 'POST', '/confirm', [1])[0] == 400
    assert request(viewer, 'POST', '/confirm', {'token': 'x'}, {'Origin': 'http://example.com'})[0] == 403

def test_confirmation_can_only_be_used_once(viewer, monkeypatch):
    async def run():
        monkeypatch.setattr(app, '_main_loop', asyncio.get_running_loop())
        monkeypatch.setattr(app, '_confirm_event', asyncio.Event())
        monkeypatch.setattr(app, '_pending_approval', {'token': 'test-token', 'round': 2})
        status, body = await asyncio.to_thread(request, viewer, 'GET', '/status')
        assert json.loads(body)['approval']['round'] == 2
        status, _ = await asyncio.to_thread(request, viewer, 'POST', '/confirm', {'token': 'test-token'})
        assert status == 200
        await asyncio.wait_for(app._confirm_event.wait(), 1)
        status, _ = await asyncio.to_thread(request, viewer, 'POST', '/confirm', {'token': 'test-token'})
        assert status == 409
    asyncio.run(run())
