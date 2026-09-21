use serde::Serialize;
use std::collections::VecDeque;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct C310IssueQueueSnapshot {
    pub capacity: usize,
    pub depth: usize,
}

impl C310IssueQueueSnapshot {
    pub const fn is_full(self) -> bool {
        self.depth == self.capacity
    }

    pub const fn is_empty(self) -> bool {
        self.depth == 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct C310IssueQueueTransition {
    pub before: C310IssueQueueSnapshot,
    pub after: C310IssueQueueSnapshot,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum C310DequeueOutcome<T> {
    Empty,
    NotReady,
    Dequeued {
        item: T,
        transition: C310IssueQueueTransition,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum C310IssueQueueError {
    #[error("issue queue capacity must be positive")]
    ZeroCapacity,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C310IssueQueue<T> {
    capacity: usize,
    entries: VecDeque<T>,
}

impl<T> C310IssueQueue<T> {
    pub fn new(capacity: usize) -> Result<Self, C310IssueQueueError> {
        if capacity == 0 {
            return Err(C310IssueQueueError::ZeroCapacity);
        }
        Ok(Self {
            capacity,
            entries: VecDeque::new(),
        })
    }

    pub fn snapshot(&self) -> C310IssueQueueSnapshot {
        C310IssueQueueSnapshot {
            capacity: self.capacity,
            depth: self.entries.len(),
        }
    }

    pub fn front(&self) -> Option<&T> {
        self.entries.front()
    }

    pub fn try_enqueue(&mut self, item: T) -> Result<C310IssueQueueTransition, T> {
        let before = self.snapshot();
        if before.is_full() {
            return Err(item);
        }
        self.entries.push_back(item);
        Ok(C310IssueQueueTransition {
            before,
            after: self.snapshot(),
        })
    }

    pub fn try_dequeue_if(&mut self, ready: impl FnOnce(&T) -> bool) -> C310DequeueOutcome<T> {
        let Some(front) = self.entries.front() else {
            return C310DequeueOutcome::Empty;
        };
        if !ready(front) {
            return C310DequeueOutcome::NotReady;
        }
        let before = self.snapshot();
        let item = self.entries.pop_front().expect("front item is present");
        C310DequeueOutcome::Dequeued {
            item,
            transition: C310IssueQueueTransition {
                before,
                after: self.snapshot(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_capacity() {
        assert_eq!(
            C310IssueQueue::<u8>::new(0),
            Err(C310IssueQueueError::ZeroCapacity)
        );
    }

    #[test]
    fn preserves_fifo_order_and_reports_transitions() {
        let mut queue = C310IssueQueue::new(2).unwrap();
        assert_eq!(queue.snapshot().depth, 0);
        let first = queue.try_enqueue(7).unwrap();
        assert_eq!((first.before.depth, first.after.depth), (0, 1));
        let second = queue.try_enqueue(9).unwrap();
        assert_eq!((second.before.depth, second.after.depth), (1, 2));
        assert!(queue.snapshot().is_full());
        assert_eq!(queue.front(), Some(&7));
        assert!(matches!(
            queue.try_dequeue_if(|_| true),
            C310DequeueOutcome::Dequeued {
                item: 7,
                transition: C310IssueQueueTransition {
                    before: C310IssueQueueSnapshot { depth: 2, .. },
                    after: C310IssueQueueSnapshot { depth: 1, .. },
                },
            }
        ));
        assert!(matches!(
            queue.try_dequeue_if(|_| true),
            C310DequeueOutcome::Dequeued { item: 9, .. }
        ));
        assert_eq!(queue.try_dequeue_if(|_| true), C310DequeueOutcome::Empty);
    }

    #[test]
    fn full_and_not_ready_paths_preserve_entries() {
        let mut queue = C310IssueQueue::new(1).unwrap();
        queue.try_enqueue(3).unwrap();
        let original = queue.clone();
        assert_eq!(queue.try_enqueue(4), Err(4));
        assert_eq!(queue, original);
        assert_eq!(
            queue.try_dequeue_if(|_| false),
            C310DequeueOutcome::NotReady
        );
        assert_eq!(queue, original);
    }
}
