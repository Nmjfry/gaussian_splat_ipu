/*
    Copyright (C) Mark Pupilli 2013, All rights reserved
*/
#pragma once

#include <VideoLib.h>
#include <ipu/ipu_utils.hpp>

#include <PacketComms.h>
#include <PacketSerialisation.h>

#include <chrono>
#include <cinttypes>
#include <memory>

// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

class PacketDemuxer;

/**
    Class for managing a video stream received over the muxer comms system.

    In the constructor a subscription is made to AvData packets which are
    simply enqued as they are received. Note that the server must be sending
    packets with the same name.
*/
class VideoClient {
public:
  VideoClient(PacketDemuxer &demuxer, const std::string &avPacketName)
      : m_packetOffset(0),
        m_avDataSubscription(demuxer.subscribe(
            avPacketName,
            [this](const ComPacket::ConstSharedPacket &packet) {
              m_avDataPackets.emplace(packet);
              m_totalVideoBytes += packet->getDataSize();

              ipu_utils::logger()->info(
                  "Received compressed video packet of size {}",
                  packet->getDataSize());
            })),
        m_avTimeout(0) {}

  ~VideoClient() {}

  bool initialiseVideoStream(const std::chrono::seconds &videoTimeout) {
    m_avTimeout = videoTimeout;
    resetAvTimeout();

    m_lastBandwidthCalcTime = std::chrono::steady_clock::now();

    // Create a video reader object that uses function/callback IO:
    m_videoIO.reset(new FFMpegStdFunctionIO(
        FFMpegCustomIO::ReadBuffer,
        std::bind(&VideoClient::readPacket, std::ref(*this),
                  std::placeholders::_1, std::placeholders::_2)));
    m_streamer.reset(new LibAvCapture(*m_videoIO));
    if (m_streamer->IsOpen() == false) {
      ipu_utils::logger()->info("Failed to open video stream.");
      return false;
    }

    // Get some frames so we can extract correct image dimensions:
    bool gotFrame = false;
    for (int i = 0; i < 2; ++i) {
      gotFrame = m_streamer->GetFrame();
      m_streamer->DoneFrame();
    }

    if (gotFrame == false) {
      ipu_utils::logger()->info("Failed to acquire frames from video.");
      return false;
    }

    auto w = getFrameWidth();
    auto h = getFrameHeight();
    ipu_utils::logger()->info("Successfully initialised video stream: {} x {}",
                              w, h);
    return true;
  }

  int getFrameWidth() const { return m_streamer->GetFrameWidth(); };
  int getFrameHeight() const { return m_streamer->GetFrameHeight(); };

  bool receiveVideoFrame(std::function<void(LibAvCapture &)> callback) {
    if (m_streamer == nullptr) {
      throw std::logic_error(std::string(__FUNCTION__) +
                             ": streamer object not allocated.");
    }

    bool gotFrame = m_streamer->GetFrame();
    if (gotFrame) {
      callback(*m_streamer);
      m_streamer->DoneFrame();
    }

    return gotFrame;
  }

  double computeVideoBandwidthConsumed() {
    std::chrono::steady_clock::time_point timeNow =
        std::chrono::steady_clock::now();
    double seconds = std::chrono::duration_cast<std::chrono::milliseconds>(
                         timeNow - m_lastBandwidthCalcTime)
                         .count() /
                     1000.0;
    double bits_per_sec =
        (m_totalVideoBytes - m_lastTotalVideoBytes) * (8.0 / seconds);
    m_lastTotalVideoBytes = m_totalVideoBytes;
    m_lastBandwidthCalcTime = timeNow;
    return bits_per_sec;
  }

  void decodeVideoFrame(std::vector<std::uint8_t> &bgrBuffer) {
    newFrameDecoded =
        receiveVideoFrame([&bgrBuffer, this](LibAvCapture &stream) {
          ipu_utils::logger()->info("Decoded video frame");
          auto w = stream.GetFrameWidth();
          stream.ExtractRgbaImage(bgrBuffer.data(), w * 4);
        });

    if (newFrameDecoded) {
      ipu_utils::logger()->info("Successfully decoded video frame");
    } else {
      ipu_utils::logger()->info("Failed to decode video frame, buffer is null.");
    }
  }

protected:
  bool streamerOk() const {
    return m_streamer != nullptr && m_streamer->IoError() == false;
  }

  bool streamerIoError() const {
    if (m_streamer.get() == nullptr) {
      ipu_utils::logger()->info("Stream capture object not allocated.");
      return false; // Streamer not allocated yet (obviosuly this does not
                    // count as IO error)
    }

    return m_streamer->IoError();
  }

  int readPacket(uint8_t *buffer, int size) {
    using namespace std::chrono_literals;
    const auto retries = 4u;
    SimpleQueue::LockedQueue lockedQueue = m_avDataPackets.lock();
    while (m_avDataPackets.empty() && m_avDataSubscription.getDemuxer().ok()) {
      lockedQueue.waitNotEmpty(1s);

      if (m_avDataPackets.empty()) {
        for (auto retry = 0u; retry < retries; retry++) {
          ipu_utils::logger()->info(
              "VideoClient timed out waiting for an AV packet. Retry {} / {}",
              retry + 1, retries);
          lockedQueue.waitNotEmpty(2s); // wait a bit longer before giving up...
          if (!m_avDataPackets.empty()) {
            ipu_utils::logger()->info("Retry {} successful", retry + 1);
            break;
          }
        }

        if (avHasTimedOut()) {
          ipu_utils::logger()->info(
              "VideoClient timed out waiting for an AV packet.");
          return -1;
        }
      }
    }

    resetAvTimeout();

    // We were asked for more than packet contains so loop through packets
    // until we have returned what we needed or there are no more packets:
    int required = size;
    while (required > 0 && m_avDataPackets.empty() == false) {
      const ComPacket::ConstSharedPacket packet = m_avDataPackets.front();
      const int availableSize = packet->getData().size() - m_packetOffset;

      if (availableSize <= required) {
        // Current packet contains less than required so copy the whole
        // packet and continue:
        std::copy(packet->getData().begin() + m_packetOffset,
                  packet->getData().end(), buffer);
        m_packetOffset = 0; // Reset the packet offset so the next packet
                            // will be read from beginning.
        m_avDataPackets.pop();
        buffer += availableSize;
        required -= availableSize;
      } else {
        // Current packet contains more than enough to fulfill the request
        // so copy what is required and save the rest for later:
        auto startItr = packet->getData().begin() + m_packetOffset;
        std::copy(startItr, startItr + required, buffer);
        m_packetOffset += required; // Increment the packet offset by the
                                    // amount read from this packet.
        required = 0;
      }
    }

    return size - required;
  }

private:
  SimpleQueue m_avInfoPackets;
  SimpleQueue m_avDataPackets;
  int m_packetOffset;
  uint64_t m_lastTotalVideoBytes;
  uint64_t m_totalVideoBytes;
  PacketSubscription m_avDataSubscription;

  std::unique_ptr<FFMpegStdFunctionIO> m_videoIO;
  std::unique_ptr<LibAvCapture> m_streamer;

  std::atomic<bool> newFrameDecoded;

  void resetAvTimeout() {
    m_avDataTimeoutPoint = std::chrono::steady_clock::now() + m_avTimeout;
  }
  bool avHasTimedOut() {
    return std::chrono::steady_clock::now() > m_avDataTimeoutPoint;
  }
  std::chrono::steady_clock::time_point m_avDataTimeoutPoint;
  std::chrono::seconds m_avTimeout;
  std::chrono::steady_clock::time_point m_lastBandwidthCalcTime;
};
