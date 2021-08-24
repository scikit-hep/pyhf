import pyhf.events as events
from unittest import mock


def test_subscribe_event():
    ename = 'test'

    m = mock.Mock()
    events.subscribe(ename)(m.__call__)
    assert ename in events.__events
    assert m.__call__.__func__ == events.__events.get(ename)[0][0]()
    del events.__events[ename]


def test_event():
    ename = 'test'

    m = mock.Mock()
    events.subscribe(ename)(m.__call__)

    events.trigger(ename)()
    m.assert_called_once()
    del events.__events[ename]


def test_event_weakref():
    ename = 'test'

    m = mock.Mock()
    events.subscribe(ename)(m.__call__)
    assert len(events.trigger(ename)) == 1
    # should be weakly referenced
    del m
    assert len(events.trigger(ename)) == 0
    del events.__events[ename]


def test_disable_event():
    ename = 'test'

    m = mock.Mock()
    noop, noop_m = events.noop, mock.Mock()
    events.noop = noop_m
    events.subscribe(ename)(m.__call__)

    events.disable(ename)
    assert m.called is False
    assert ename in events.__disabled_events
    assert events.trigger(ename) == events.noop
    assert events.trigger(ename)() == events.noop()
    assert m.called is False
    assert noop_m.is_called_once()
    events.enable(ename)
    assert ename not in events.__disabled_events
    del events.__events[ename]
    events.noop = noop


def test_trigger_noevent():
    noop, noop_m = events.noop, mock.Mock()

    assert 'fake' not in events.__events
    assert events.trigger('fake') == events.noop
    assert events.trigger('fake')() == events.noop()
    assert noop_m.is_called_once()

    events.noop = noop


def test_subscribe_function(capsys):
    ename = 'test'

    def add(a, b):
        print(a + b)

    events.subscribe(ename)(add)
    events.trigger(ename)(1, 2)

    captured = capsys.readouterr()
    assert captured.out == "3\n"

    del events.__events[ename]


def test_trigger_function(capsys):
    ename = 'test'

    def add(a, b):
        print(a + b)

    precall = mock.Mock()
    postcall = mock.Mock()

    wrapped_add = events.register(ename)(add)
    events.subscribe(f'{ename}::before')(precall.__call__)
    events.subscribe(f'{ename}::after')(postcall.__call__)

    precall.assert_not_called()
    postcall.assert_not_called()

    wrapped_add(1, 2)
    captured = capsys.readouterr()
    assert captured.out == "3\n"
    precall.assert_called_once()
    postcall.assert_called_once()

    del events.__events[f'{ename}::before']
    del events.__events[f'{ename}::after']
