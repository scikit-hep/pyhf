import pyhf.events as events
import mock


def test_subscribe_event():
    ename = 'test'

    m = mock.Mock()
    events.subscribe(ename)(m.__call__)

    assert ename in events.__events
    assert m.__call__ == events.__events.get(ename)[0]()
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
