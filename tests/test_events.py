import pyhf.events as events
import mock


def test_subscribe_event():
    ename = 'test'

    m = mock.Mock()
    events.subscribe(ename)(m)

    assert ename in events.__events
    assert m in events.__events.get(ename)
    del events.__events[ename]


def test_event():
    ename = 'test'

    m = mock.Mock()
    events.subscribe(ename)(m)

    events.trigger(ename)()
    m.assert_called_once()
    del events.__events[ename]


def test_disable_event():
    ename = 'test'

    m = mock.Mock()
    noop, noop_m = events.noop, mock.Mock()
    events.noop = noop_m
    events.subscribe(ename)(m)

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
