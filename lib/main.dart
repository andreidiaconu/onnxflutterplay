import 'dart:async';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const MyHomePage(title: 'Flutter Demo Home Page'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  int _counter = 0;
  ui.Image? image;

  void _incrementCounter() {
    // _inferSingleAdd();
    _inferMosaic9();
    setState(() {
      _counter++;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: Text(widget.title),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Image.asset('assets/bluetit.jpg'),
            RawImage(
              image: image,
            )
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _incrementCounter,
        tooltip: 'Run ONNX model',
        child: const Icon(Icons.play_arrow),
      ),
    );
  }

  void _inferSingleAdd() async {
    OrtEnv.instance.init();
    final sessionOptions = OrtSessionOptions();
    final rawAssetFile = await rootBundle.load("assets/models/single_add.ort");
    final bytes = rawAssetFile.buffer.asUint8List();
    final session = OrtSession.fromBuffer(bytes, sessionOptions);
    final runOptions = OrtRunOptions();
    final inputOrt = OrtValueTensor.createTensorWithDataList(
        Float32List.fromList([5.9]),
    );
    final inputs = {'A':inputOrt, 'B': inputOrt};
    final outputs = session.run(runOptions, inputs);
    inputOrt.release();
    runOptions.release();
    sessionOptions.release();
    // session.release();
    OrtEnv.instance.release();
    List c = outputs[0]?.value as List;
    print(c[0] ?? "none");
  }

  void _inferMosaic9() async {
    OrtEnv.instance.init();
    final sessionOptions = OrtSessionOptions();
    // You can also try pointilism-9.ort and rain-princess.ort
    final rawAssetFile = await rootBundle.load("assets/models/mosaic-9.ort");
    final bytes = rawAssetFile.buffer.asUint8List();
    final session = OrtSession.fromBuffer(bytes, sessionOptions);
    final runOptions = OrtRunOptions();

    // You can also try red.png, redgreen.png, redgreenblueblack.png for easy debug
    ByteData blissBytes = await rootBundle.load('assets/bluetit.jpg');
    final image = await decodeImageFromList(Uint8List.sublistView(blissBytes));
    final rgbFloats = await imageToFloatTensor(image);
    final inputOrt = OrtValueTensor.createTensorWithDataList(Float32List.fromList(rgbFloats), [1, 3, 224, 224]);

    final inputs = {'input1':inputOrt};
    final outputs = session.run(runOptions, inputs);
    inputOrt.release();
    runOptions.release();
    sessionOptions.release();
    // session.release();
    OrtEnv.instance.release();
    List outFloats = outputs[0]?.value as List;
    print(outFloats[0] ?? "none");

    final result = await floatTensorToImage(outFloats);
    setState(() {
      this.image = result;
    });
  }

  Future<List<double>> imageToFloatTensor(ui.Image image) async {
    final imageAsFloatBytes = (await image.toByteData(format: ui.ImageByteFormat.rawRgba))!;
    final rgbaUints = Uint8List.view(imageAsFloatBytes.buffer);

    final indexed = rgbaUints.indexed;
    return [
    ...indexed.where((e) => e.$1 % 4 == 0).map((e) => e.$2.toDouble()),
    ...indexed.where((e) => e.$1 % 4 == 1).map((e) => e.$2.toDouble()),
    ...indexed.where((e) => e.$1 % 4 == 2).map((e) => e.$2.toDouble()),
    ];
  }

  Future<ui.Image> floatTensorToImage(List tensorData) {
    final outRgbaFloats = Uint8List(4 * 224 * 224);
    for (int x = 0; x < 224; x++) {
      for (int y = 0; y < 224; y++) {
        final index = x * 224 * 4 + y * 4;
        outRgbaFloats[index + 0] = tensorData[0][0][x][y].clamp(0, 255).toInt(); // r
        outRgbaFloats[index + 1] = tensorData[0][1][x][y].clamp(0, 255).toInt(); // g
        outRgbaFloats[index + 2] = tensorData[0][2][x][y].clamp(0, 255).toInt(); // b
        outRgbaFloats[index + 3] = 255; // a
      }
    }
    final completer = Completer<ui.Image>();
    ui.decodeImageFromPixels(outRgbaFloats, 224, 224, ui.PixelFormat.rgba8888, (ui.Image image) {
      completer.complete(image);
    });

    return completer.future;
  }
}
