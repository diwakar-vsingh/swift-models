// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Datasets
import ModelSupport
import XCTest

class WordSegDatasetTests: XCTestCase {
  func testCreateWordSegDatasetReference() {
    do {
      let dataset = try WordSegDataset()
      XCTAssertEqual(dataset.trainingPhrases.count, 7832)
      XCTAssertEqual(dataset.validationPhrases.count, 979)
      XCTAssertEqual(dataset.testingPhrases.count, 979)

      // Check the first example in each set.
      let trainingExample: [Int32] = [
        26, 16, 22, 24, 2, 15, 21, 21, 16, 20, 6, 6, 21,
        9, 6, 3, 16, 16, 12, 28,
      ]
      XCTAssertEqual(dataset.trainingPhrases[0].numericalizedText.characters, trainingExample)

      let validationExample: [Int32] = [9, 6, 13, 13, 16, 14, 10, 14, 10, 28]
      XCTAssertEqual(dataset.validationPhrases[0].numericalizedText.characters, validationExample)

      let testingExample: [Int32] = [
        13, 6, 21, 14, 6, 20, 6, 6, 10, 7, 10, 4,
        2, 15, 20, 6, 6, 2, 15, 26, 3, 16, 5, 26, 10, 15, 21, 9, 2, 21, 14, 10,
        19, 19, 16, 19, 28,
      ]
      XCTAssertEqual(dataset.testingPhrases[0].numericalizedText.characters, testingExample)
    } catch {
      XCTFail(error.localizedDescription)
    }
  }

  func testCreateWordSegDatasetTrainingOnly() {
    do {
      let localStorageDirectory: URL = DatasetUtilities.defaultDirectory
        .appendingPathComponent("WordSeg", isDirectory: true)
      let trainingFile = localStorageDirectory.appendingPathComponent("/seg/br/br-text/tr.txt")
      let dataset = try WordSegDataset(training: trainingFile.path)
      XCTAssertEqual(dataset.trainingPhrases.count, 7832)
      XCTAssertEqual(dataset.validationPhrases.count, 0)
      XCTAssertEqual(dataset.testingPhrases.count, 0)

      // Check the first example in each set.
      let trainingExample: [Int32] = [
        26, 16, 22, 24, 2, 15, 21, 21, 16, 20, 6, 6, 21,
        9, 6, 3, 16, 16, 12, 28,
      ]
      XCTAssertEqual(dataset.trainingPhrases[0].numericalizedText.characters, trainingExample)
    } catch {
      XCTFail(error.localizedDescription)
    }
  }

  func testWordSegDatasetLoad() {
    let buffer: [UInt8] = [
      0x61, 0x6c, 0x70, 0x68, 0x61, 0x0a,  // alpha.
    ]

    var dataset: WordSegDataset?
    buffer.withUnsafeBytes { pointer in
      guard let address = pointer.baseAddress else { return }
      let training: Data =
        Data(
          bytesNoCopy: UnsafeMutableRawPointer(mutating: address),
          count: pointer.count, deallocator: .none)
      dataset = WordSegDataset(training: training, validation: nil, testing: nil)
    }

    // 'a', 'h', 'l', 'p', '</s>', '</w>', '<pad>'
    XCTAssertEqual(dataset?.alphabet.count, 7)
    XCTAssertEqual(dataset?.trainingPhrases.count, 1)
  }

  static var allTests = [
    ("testCreateWordSegDatasetReference", testCreateWordSegDatasetReference),
    ("testCreateWordSegDatasetTrainingOnly", testCreateWordSegDatasetTrainingOnly),
    ("testWordSegDatasetLoad", testWordSegDatasetLoad),
  ]
}
